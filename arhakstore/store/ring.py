"""ShardRing: many blobs sharing one pool of RS shard files.

Per-entry vaults (store.vault.Vault) write n shard files per blob, so N
blobs cost 48N files. A ring stores all blobs' symbols for position j in one
shared file, so the whole archive costs O(n) files regardless of blob count:

    ring_shard_0000.bin  =  [golay header][ col_0(blob0) ][ col_0(blob1) ]...
    ring_shard_0001.bin  =  [golay header][ col_1(blob0) ][ col_1(blob1) ]...
    ...

Position j of blob b's chunk c is the RS symbol j of that chunk. Any k of the
n positions rebuild a blob exactly; `kill_positions` simulates whole shard
files dying; `heal` regenerates the dead columns from k surviving positions.

Layout invariants (prototype, honest):
  * Blob order in the manifest matches segment order in every ring file.
  * add_bytes appends columns; heal/delete rewrite the affected ring files.
  * Each blob keeps per-position BLAKE2b column digests for live admission.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ._io import atomic_write_text
from .gf256 import GF256
from .golay24 import GolayCode
from .rs import ReedSolomon, PROFILES
from .shards import b2b, parse_shard_header, shard_header

RING_VERSION = 1
DEFAULT_PROFILE = "quadallic"
_COL_DIGEST_BYTES = 16


class RingError(Exception):
    pass


def _column_digest(data: bytes) -> str:
    return b2b(data, _COL_DIGEST_BYTES)


class ShardRing:
    """A shared-pool erasure archive for many blobs."""

    def __init__(self, root, profile: str = DEFAULT_PROFILE):
        if profile not in PROFILES:
            raise RingError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")
        self.root = Path(root)
        self.shard_dir = self.root / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "ring.json"
        self.profile = profile
        self._golay = GolayCode()
        self._codec = None
        self._manifest = self._load_or_initial()

    # ------------------------------------------------------------- manifest
    def _empty_manifest(self) -> dict:
        n, k, _ = PROFILES[self.profile]
        return {
            "version": RING_VERSION,
            "profile": self.profile,
            "n": n,
            "k": k,
            "dead": [],
            "blobs": [],
        }

    def _load_or_initial(self) -> dict:
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if data.get("version") != RING_VERSION:
                raise RingError(f"unsupported ring version {data.get('version')}")
            return data
        return self._empty_manifest()

    def _save(self) -> None:
        atomic_write_text(self.manifest_path, json.dumps(self._manifest, indent=2,
                                                         sort_keys=True))

    # -------------------------------------------------------------- helpers
    @property
    def n(self) -> int:
        return self._manifest["n"]

    @property
    def k(self) -> int:
        return self._manifest["k"]

    @property
    def blobs(self) -> list:
        return self._manifest["blobs"]

    def _rs(self):
        if self._codec is None:
            self._codec = ReedSolomon(self.n, self.k)
        return self._codec

    def _pos_file(self, j: int) -> Path:
        return self.shard_dir / f"ring_shard_{j:04d}.bin"

    def _bounds(self) -> list:
        """Cumulative segment boundaries over blob chunk counts."""
        out = []
        acc = 0
        for b in self.blobs:
            out.append((acc, acc + b["chunks"]))
            acc += b["chunks"]
        return out

    def _chunks_of(self, data: bytes) -> list:
        k = self.k
        chunks = []
        for i in range(0, len(data), k):
            chunk = data[i : i + k]
            if len(chunk) < k:
                chunk += bytes(k - len(chunk))
            chunks.append(chunk)
        if not chunks:
            chunks.append(bytes(k))
        return chunks

    def _pad_of(self, data: bytes) -> int:
        k = self.k
        return k - (len(data) % k) if len(data) % k else 0

    def _split_positions(self, data: bytes) -> tuple:
        """data -> (chunks, pad, stripes) where stripes[j] is the length-c chunk
        column for position j."""
        rs = self._rs()
        chunks = self._chunks_of(data)
        stripes = [bytearray() for _ in range(self.n)]
        for chunk in chunks:
            code = rs.encode(chunk)
            for j in range(self.n):
                stripes[j].append(code[j])
        return chunks, self._pad_of(data), stripes

    # ----------------------------------------------------------------- add
    def add_bytes(self, data: bytes, name: str | None = None, dedupe: bool = True) -> int:
        """Register a blob; returns its id (index in the manifest)."""
        digest = b2b(data, 32)
        if dedupe:
            for b in self.blobs:
                if b["file_digest"] == digest:
                    return self.blobs.index(b)
        chunks, _, stripes = self._split_positions(data)
        blob_id = len(self.blobs)
        self.blobs.append(
            {
                "name": name or f"blob-{blob_id:05d}",
                "size": len(data),
                "pad": self._pad_of(data),
                "chunks": len(chunks),
                "file_digest": digest,
                "col_digests": {},
            }
        )
        for j in range(self.n):
            path = self._pos_file(j)
            if not path.exists():
                path.write_bytes(shard_header(j, self._golay))
            with path.open("ab") as fh:
                fh.write(bytes(stripes[j]))
            self.blobs[blob_id]["col_digests"][str(j)] = _column_digest(bytes(stripes[j]))
        self._save()
        return blob_id

    def add(self, path, name: str | None = None) -> int:
        return self.add_bytes(Path(path).read_bytes(), name=name or Path(path).name)

    # ---------------------------------------------------------------- read
    def _position_segments(self):
        """{j: [column(blob0), column(blob1), ...]} for usable position files."""
        if not self.blobs:
            return {}  # empty ring: nothing segmented yet
        bounds = self._bounds()
        out = {}
        for j in range(self.n):
            path = self._pos_file(j)
            if not path.exists():
                continue
            blob = path.read_bytes()
            try:
                idx = parse_shard_header(blob[:3], self._golay)
            except ValueError:
                continue
            if idx != j:
                continue
            body = blob[3:]
            segs = []
            ok = True
            for s, e in bounds:
                if e > len(body):
                    ok = False
                    break
                segs.append(body[s:e])
            if not ok or len(body) > bounds[-1][1]:
                continue
            out[j] = segs
        return out

    def _verified_columns(self):
        """(per_blob[j]->bytes verified, corrupt[{b}->set(j)], missing[{b}->set(j)]).

        Whole-position losses hit every blob; digest failures hit only the
        blob whose column drifted.
        """
        dead = set(self._manifest["dead"])
        positions = self._position_segments()
        nb = len(self.blobs)
        verified: dict = {b: {} for b in range(nb)}
        corrupt: dict = {b: set() for b in range(nb)}
        missing: dict = {b: set() for b in range(nb)}
        for j in range(self.n):
            in_dead: list = []
            if j in dead or j not in positions:
                if j in dead:
                    for b in range(nb):
                        missing[b].add(j)
                continue
            segs = positions[j]
            for b in range(nb):
                if len(segs) > b:
                    col = segs[b]
                    if b2b(col, _COL_DIGEST_BYTES) != self.blobs[b]["col_digests"][str(j)]:
                        corrupt[b].add(j)
                    else:
                        verified[b][j] = col
        return verified, corrupt, missing

    def read_bytes(self, blob_id: int) -> bytes:
        """Rebuild blob `blob_id` from any k verified (or correctable) columns."""
        nb = len(self.blobs)
        if not 0 <= blob_id < nb:
            raise RingError(f"unknown blob id {blob_id}")
        verified, corrupt, missing = self._verified_columns()
        if len(verified[blob_id]) < self.k:
            raise RingError(
                f"blob {blob_id}: need {self.k} intact columns, have "
                f"{len(verified[blob_id])} (run heal)"
            )
        b = self.blobs[blob_id]
        rs = self._rs()
        r = self.n - self.k  # parity prefix -> message occupies codeword[r:]
        positions = list(verified[blob_id])[: self.k]
        repairer = rs.repairer(positions)
        want = list(range(r, self.n))  # k systematic (message) symbols
        chunks = b["chunks"]
        out = bytearray()
        for c in range(chunks):
            survivors = {p: verified[blob_id][p][c] for p in positions}
            out += bytes(repairer.recover_symbols(survivors, want))
        data = bytes(out)[: b["size"]]
        if b2b(data, 32) != b["file_digest"]:
            raise RingError(f"blob {blob_id}: reconstructed digest mismatch")
        return data

    def read(self, blob_id: int, out) -> str:
        data = self.read_bytes(blob_id)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        verified, corrupt, missing = self._verified_columns()
        dead = set(self._manifest["dead"])
        rows = []
        total_ok = total_columns = 0
        for b in range(len(self.blobs)):
            blob = self.blobs[b]
            ok, corr, miss = len(verified[b]), len(corrupt[b]), len(missing[b])
            total_ok += ok
            total_columns += self.n  # per-blob column positions (one row per blob * n)
            rows.append({
                "id": b,
                "name": blob["name"],
                "size": blob["size"],
                "chunks": blob["chunks"],
                "ok": ok,
                "corrupt": corr,
                "missing": miss,
                "loss_budget": self.n - self.k,
                "surplus": ok - self.k,
                "healable": ok >= self.k,
            })
        return {
            "profile": self.profile,
            "n": self.n,
            "k": self.k,
            "blobs": len(rows),
            "dead_positions": sorted(dead),
            "columns_ok": total_ok,
            "columns_total": total_columns,
            "blob_status": rows,
        }

    # ------------------------------------------------------------ damage/heal
    def kill_positions(self, positions) -> list:
        dead = set(self._manifest["dead"])
        for p in positions:
            if 0 <= p < self.n:
                dead.add(p)
        self._manifest["dead"] = sorted(dead)
        self._save()
        return sorted(dead)

    def _rebuild_column(self, b: int, j: int, verified: dict, repairer, rs) -> bytes:
        chunks = self.blobs[b]["chunks"]
        positions = list(verified[b])[: self.k]
        out = bytearray()
        for c in range(chunks):
            survivors = {p: verified[b][p][c] for p in positions}
            out.append(repairer.recover_symbols(survivors, [j])[0])
        return bytes(out)

    def heal(self) -> list:
        """Regenerate every dead/corrupt column for every blob; returns
        (blob, position) pairs rebuilt."""
        verified, corrupt, missing = self._verified_columns()
        rebuilt = []
        needs = {b: set(self._manifest["dead"]) | corrupt[b] | missing[b]
                 for b in range(len(self.blobs))}
        for b, cols in needs.items():
            if len(verified[b]) < self.k:
                raise RingError(
                    f"blob {b}: only {len(verified[b])}/{self.k} intact columns; cannot heal"
                )
        rs = self._rs()
        repairers = {}
        for b in range(len(self.blobs)):
            if not needs[b]:
                continue
            positions = list(verified[b])[: self.k]
            repairers[b] = rs.repairer(positions)
            for j in needs[b]:
                verified[b][j] = self._rebuild_column(b, j, verified, repairers[b], rs)
                rebuilt.append((b, j))
        self._rewrite_all(verified)
        self._manifest["dead"] = []
        self._save()
        return rebuilt

    def _rewrite_all(self, columns: dict) -> None:
        nb = len(self.blobs)
        for j in range(self.n):
            body = bytearray()
            ok = True
            for b in range(nb):
                col = columns.get(b, {}).get(j)
                if col is None:
                    ok = False
                    break
                body += col
            if not ok:
                continue
            self._pos_file(j).write_bytes(shard_header(j, self._golay) + bytes(body))

    # -------------------------------------------------------------- delete
    def delete(self, blob_id: int) -> None:
        nb = len(self.blobs)
        if not 0 <= blob_id < nb:
            raise RingError(f"unknown blob id {blob_id}")
        # Capture the per-blob segments with the CURRENT layout before any
        # file rewrite; then drop the blob and write the files back shorter.
        positions = self._position_segments()
        if not positions:
            raise RingError("position files unreadable; cannot delete")
        keep = [b for b in range(nb) if b != blob_id]
        del self.blobs[blob_id]
        nkeep = len(keep)
        for j, segs in positions.items():
            if len(segs) < nb:
                continue
            body = bytearray(shard_header(j, self._golay))
            for b in keep:
                body += segs[b]
            self._pos_file(j).write_bytes(bytes(body))
        self._save()

    def destroy(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)