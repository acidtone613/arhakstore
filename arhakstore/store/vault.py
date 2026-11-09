"""Vault: self-healing redundant file store built on the DMF RS tower.

Design (the "something useful + new" pitch):

  * A blob is fanned out into n shard files with a systematic Reed-Solomon
    code over GF(2^8). Any k of the n shards reconstruct the blob exactly.
    The default "engine" profile is the corpus's own RS(253, 221, 33):
    ~14% overhead buys survival of any 32 lost/corrupt shards of 253.

  * Every shard carries a Golay-protected 3-byte header (magic + 12-bit
    index), so a shard self-identifies and self-repairs small header
    corruption (t = 3) — the memory-addressing motif from OC-RC06.

  * `heal()` regenerates lost shards from k surviving neighbors — a
    deterministic erasure "regeneration" of the OC-RC08/09 cellular-automaton
    motif: any k live cells rebuild a dead one exactly.

  * Integrity is a Merkle-ish structure: per-shard BLAKE2b digests in the
    manifest; a corrupt shard is demoted to "lost" and healed from the rest.

  * `decode_bytes()` runs the full Berlekamp-Massey + Forney decoder over
    every chunk: any pattern with 2*nu + mu <= n-k (random symbol errors in
    surviving shards + whole-shard erasures) is corrected exactly — recovery
    even when a corrupt shard was never flagged.

Honest limits (kept, not hidden):
  * Prototype is per-blob (fine to ~100 MB); no streaming yet.
  * Reconstructed data is only ever trusted if it matches the manifest's
    whole-file BLAKE2b digest.
  * k x k GF matrix inversion is O(k^3); the engine profile recomputes it
    once per restore/heal (seconds), fine for a prototype.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from . import shards as shards_mod
from ._io import atomic_write_text
from .golay24 import GolayCode
from .rs import DecodeError, ReedSolomon, PROFILES
from .shards import Manifest, b2b, parse_shard_header, shard_header

DEFAULT_PROFILE = "engine"


class VaultError(Exception):
    pass


class Vault:
    """A single-vault redundant store rooted at `root`."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.shard_dir = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"
        self._golay = GolayCode()

    # ------------------------------------------------------------------- write
    def archive(self, path: str | os.PathLike, profile: str = DEFAULT_PROFILE) -> str:
        """Fan a file out into n shard files with a manifest."""
        path = Path(path)
        return self.archive_bytes(path.read_bytes(), name=path.name, profile=profile)

    def archive_bytes(
        self, data: bytes, name: str = "blob", profile: str = DEFAULT_PROFILE
    ) -> str:
        """Fan an in-memory blob out into n shard files with a manifest."""
        if profile not in PROFILES:
            raise VaultError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")
        n, k, _ = PROFILES[profile]
        rs = ReedSolomon(n, k)
        chunks, pad = self._split_chunks(data, k)

        stripes = [bytearray() for _ in range(n)]
        for chunk in chunks:
            code = rs.encode(chunk)
            for j in range(n):
                stripes[j].append(code[j])

        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        digests: dict = {}
        for j in range(n):
            body = bytes(stripes[j])
            file = self.shard_dir / self._shard_name(j)
            file.write_bytes(shard_header(j, self._golay) + body)
            digests[j] = b2b(body, 32)

        manifest = Manifest(
            name=name,
            profile=profile,
            n=n,
            k=k,
            size=len(data),
            pad=pad,
            chunks=len(chunks),
            file_digest=b2b(data, 32),
            shard_digests=digests,
        )
        atomic_write_text(self.manifest_path, manifest.to_json())
        return str(self.manifest_path)

    @staticmethod
    def _split_chunks(data: bytes, k: int):
        chunks = []
        for i in range(0, len(data), k):
            chunk = data[i : i + k]
            if len(chunk) < k:
                chunk += bytes(k - len(chunk))
            chunks.append(chunk)
        if not chunks:
            chunks.append(bytes(k))
        pad = k - (len(data) % k) if len(data) % k else 0
        return chunks, pad

    @staticmethod
    def _shard_name(j: int) -> str:
        return f"shard_{j:03d}.bin"

    # ------------------------------------------------------------------- read
    def load(self) -> tuple:
        if not self.manifest_path.exists():
            raise VaultError(f"no manifest at {self.manifest_path}")
        manifest = Manifest.from_json(self.manifest_path.read_text(encoding="utf-8"))
        return manifest, ReedSolomon(manifest.n, manifest.k)

    def _shard_file(self, j: int):
        return self.shard_dir / self._shard_name(j)

    def _read_all_shards(self) -> tuple:
        """Return (manifest, positions) where positions[j] is body bytes or
        None when the shard is unusable (missing or header uncorrectable)."""
        manifest, _ = self.load()
        positions = []
        for j in range(manifest.n):
            file = self._shard_file(j)
            if not file.exists():
                positions.append(None)
                continue
            blob = file.read_bytes()
            try:
                idx = parse_shard_header(blob[:3], self._golay)
            except ValueError:
                positions.append(None)
                continue
            positions.append(blob[3:] if idx == j else None)
        return manifest, positions

    def decode_bytes(self, digest_demotion: bool = True) -> bytes:
        """Errors + erasures decode of the whole blob.

        Reads every present shard; whole-shard losses and uncorrectable
        headers are erasures. Symbol drift inside a present shard is either
        handed to the decoder as random errors (digest_demotion=False) or,
        when its per-shard digest fails (digest_demotion=True), declared as
        an erasure (the cheaper, guaranteed path). Both satisfy 2*nu + mu
        <= r per chunk. The result is only returned if it matches the
        manifest's whole-file digest.
        """
        manifest, positions = self._read_all_shards()
        rs = ReedSolomon(manifest.n, manifest.k)
        r = manifest.n - manifest.k  # parity is the codeword prefix; message = [r:]
        chunks = manifest.chunks
        msg_chunks = []
        for c in range(chunks):
            received = [0] * manifest.n
            erasures = []
            for j in range(manifest.n):
                body = positions[j]
                if body is None:
                    erasures.append(j)
                    continue
                if c < len(body):
                    received[j] = body[c]
                    if digest_demotion:
                        if manifest.shard_digests.get(j) != b2b(body, 32):
                            erasures.append(j)
                else:
                    erasures.append(j)
            try:
                fixed = rs.decode_correct(received, erasures)
            except DecodeError as exc:
                raise VaultError(f"decode: chunk {c} beyond budget ({exc})") from exc
            msg_chunks.append(bytes(fixed[r:]))
        data = b"".join(msg_chunks)[: manifest.size]
        if b2b(data, 32) != manifest.file_digest:
            raise VaultError("decode: reconstructed blob digest mismatch; beyond budget?")
        return data

    def verified_positions(self):
        """Return {j: body_bytes} for shards that are present and intact."""
        manifest, _ = self.load()
        out = {}
        for j in range(manifest.n):
            file = self._shard_file(j)
            if not file.exists():
                continue
            blob = file.read_bytes()
            try:
                idx = parse_shard_header(blob[:3], self._golay)
            except ValueError:
                continue
            if idx != j:
                continue
            body = blob[3:]
            if manifest.shard_digests.get(j) != b2b(body, 32):
                continue
            out[j] = body
        return out

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        manifest, _ = self.load()
        verified = self.verified_positions()
        ok = set(verified)
        states = {"ok": [], "missing": [], "corrupt": [], "header_bad": []}
        for j in range(manifest.n):
            file = self._shard_file(j)
            if not file.exists():
                states["missing"].append(j)
                continue
            blob = file.read_bytes()
            idx = None
            try:
                idx = parse_shard_header(blob[:3], self._golay)
            except ValueError:
                pass
            if idx is None or idx != j:
                states["header_bad"].append(j)
            elif j not in ok:
                states["corrupt"].append(j)
            else:
                states["ok"].append(j)
        present = len(ok)
        return {
            "profile": manifest.profile,
            "n": manifest.n,
            "k": manifest.k,
            "chunks": manifest.chunks,
            "size": manifest.size,
            "ok": len(states["ok"]),
            "corrupt": len(states["corrupt"]),
            "header_bad": len(states["header_bad"]),
            "missing": len(states["missing"]),
            "redundant": present - manifest.k,
            "loss_budget": manifest.n - manifest.k,
            "healable": present >= manifest.k,
            "states": states,
        }

    # ---------------------------------------------------------- reconstruction
    def _build_chunks(self, verified: dict, k: int) -> list:
        """Reconstruct the k-byte chunks from any k verified shard stripes.

        The code is systematic with the message in the LAST k symbols, so the
        r = n-k parity shards plus the systematic block overlap; the fast path
        uses the systematic shards r..n-1 when they are all intact.
        """
        manifest = self.load()[0]
        n_chunks = manifest.chunks
        r = manifest.n - k
        chunks = []
        if all(j in verified for j in range(r, manifest.n)):
            # fast path: all systematic (message-bearing) shards present
            for c in range(n_chunks):
                chunks.append(bytes(verified[j][c] for j in range(r, manifest.n)))
            return chunks
        positions = list(verified)[:k]
        rs = ReedSolomon(manifest.n, k)
        repairer = rs.repairer(positions)
        for c in range(n_chunks):
            survivors = {j: verified[j][c] for j in positions}
            full = repairer.recover(survivors)
            chunks.append(full[r:])  # message symbols = codeword[r:]
        return chunks

    def extract_bytes(self) -> bytes:
        """Reconstruct the original blob from any k verified shards."""
        manifest, _ = self.load()
        verified = self.verified_positions()
        if len(verified) < manifest.k:
            raise VaultError(
                f"need {manifest.k} intact shards, have {len(verified)}; run heal or restore"
            )
        chunks = self._build_chunks(verified, manifest.k)
        data = b"".join(chunks)[: manifest.size]
        if b2b(data, 32) != manifest.file_digest:
            raise VaultError("reconstructed blob digest mismatch")
        return data

    def extract(self, out: str | os.PathLike) -> str:
        """Reconstruct the original blob from any k verified shards."""
        data = self.extract_bytes()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    # -------------------------------------------------------------------- heal
    def heal(self, want: list | None = None) -> list:
        """Regenerate every missing/corrupt shard from k surviving neighbors.

        `want` optionally restricts healing to a list of positions.
        Returns the positions that were rebuilt.
        """
        manifest, rs = self.load()
        verified = self.verified_positions()
        if len(verified) < manifest.k:
            raise VaultError(
                f"cannot heal: only {len(verified)}/{manifest.k} intact shards"
            )
        positions = list(verified)[: manifest.k]
        repairer = rs.repairer(positions)
        healed = []
        for j in range(manifest.n):
            if j in verified:
                continue
            if want is not None and j not in want:
                continue
            body = bytearray()
            for c in range(manifest.chunks):
                survivors = {p: verified[p][c] for p in positions}
                body.append(repairer.recover_symbols(survivors, [j])[0])
            self._shard_file(j).write_bytes(shard_header(j, self._golay) + bytes(body))
            healed.append(j)
        return healed

    # ------------------------------------------------------------------- misc
    def verify(self, full: bool = False) -> dict:
        """Per-shard integrity sweep.

        `full=True` also re-decodes the blob and re-hashes it against the
        manifest's whole-file digest, so `file_ok` means "recomputed digest
        matches" instead of merely "digest recorded".
        """
        st = self.status()
        manifest, _ = self.load()
        file_ok = True
        if full:
            try:
                file_ok = b2b(self.decode_bytes(), 32) == manifest.file_digest
            except (VaultError, ValueError):
                file_ok = False
        return {
            "profile": st["profile"],
            "n": st["n"],
            "k": st["k"],
            "ok": st["ok"],
            "corrupt": st["corrupt"],
            "header_bad": st["header_bad"],
            "missing": st["missing"],
            "file_ok": file_ok,
            "full": full,
        }

    def destroy(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def heal_report(self) -> list:
        st = self.status()
        if not st["healable"]:
            raise VaultError("not healable")
        missing = st["states"]["missing"] + st["states"]["corrupt"] + st["states"]["header_bad"]
        return missing