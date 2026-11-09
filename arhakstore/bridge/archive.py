"""AssociativeArchive — AR-HAK memory semantics over an exact erasure store.

The insight this merge exists to prove:

  AR-HAK heals a killed atom's *value slices* approximately, as the mean of
  its Z7 group-mates. That is the right structure for a semantic layer.

  dmfstore heals byte-exactly: any killed position of an RS codeword is
  regenerated from the surviving k positions over GF(2^8). No entropy is
  invented; the bytes come back identical.

Here the two are wired together: every memory entry keeps its AR-HAK float
bank (attention + holographic address), AND its payload is archived as a
quadallic vault whose 48 shard files ARE the 48 atoms. `kill_atoms` deletes
the corresponding shard files; `heal` regenerates them exactly; `get` is
read-through and returns the bit-exact payload even while atoms are dead.

Payload forms: `put` fans a float vector out losslessly at float32
precision (the AR-HAK semantic layer's native width); `put_bytes` stores an
arbitrary byte string verbatim (full float64 arrays via `.tobytes()`, text,
model weights, anything). Both heal bit-exactly.

Honest framing: the *semantic* layer (lookup/aggregate) is approximate by
design after heavy damage; the *archival* layer (get(entry_id)) is exact by
construction. Both claim spaces are kept visible.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..mem.layer import MemoryLayer
from ..store.vault import Vault, VaultError

QUADALLIC = "quadallic"
N_ATOMS = 48
DIM = N_ATOMS * 2  # MemoryLayer block=2 -> 96 floats per entry


def pack_value(value: np.ndarray) -> bytes:
    """Serialize a float vector losslessly (float32) for atom-fan-out."""
    return np.asarray(value, dtype=np.float32).tobytes()


def unpack_value(payload: bytes, dim: int = DIM) -> np.ndarray:
    """Deserialize the exact payload back to float64."""
    arr = np.frombuffer(payload, dtype=np.float32)
    if arr.size < dim:
        arr = np.concatenate([arr, np.zeros(dim - arr.size, dtype=np.float32)])
    return arr[:dim].astype(np.float64)


class AssociativeArchive:
    """Content-addressed, attention-retrievable, exactly self-healing memory."""

    def __init__(self, root, seed: int = 343):
        self.root = Path(root)
        self.entries_dir = self.root / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.mem = MemoryLayer(block=2, seed=seed)
        self.n_entries = 0
        self.kinds: list = []
        self._load_index()
        if (self.root / "index.json").exists():
            self.mem.load(str(self.root / "memory.npz"))

    def _load_index(self):
        path = self.root / "index.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.n_entries = data.get("n_entries", 0)
            self.kinds = data.get("kinds", ["ndarray"] * self.n_entries)
            self.kinds = (self.kinds + ["ndarray"] * self.n_entries)[: self.n_entries]

    # ------------------------------------------------------------- addressing
    def _vault(self, entry_id: int) -> Vault:
        return Vault(self.entries_dir / f"e{entry_id:05d}")

    # ---------------------------------------------------------------- write
    def put(self, key_vec, value_vec, meta=None) -> int:
        """Add one entry whose payload is a float vector (float32 fan-out).

        Returns its integer id (row index)."""
        entry_id = self.n_entries
        v = self._vault(entry_id)
        v.archive_bytes(
            pack_value(value_vec), name=f"row-{entry_id:05d}", profile=QUADALLIC
        )
        self.mem.write(
            key_vec, value_vec, meta=meta or f"row-{entry_id:05d}", bits=None
        )
        self.n_entries += 1
        self.kinds.append("ndarray")
        self._save_index()
        return entry_id

    def put_bytes(self, key_vec, payload: bytes, meta=None) -> int:
        """Add one entry whose payload is an arbitrary byte string (verbatim).

        The exact bytes — float64 arrays via `.tobytes()`, text, weights —
        are fanned out and come back bit-identical through `get_bytes`."""
        entry_id = self.n_entries
        v = self._vault(entry_id)
        v.archive_bytes(payload, name=f"row-{entry_id:05d}", profile=QUADALLIC)
        self.mem.write(
            key_vec, np.zeros(DIM), meta=meta or f"row-{entry_id:05d}", bits=None
        )
        self.n_entries += 1
        self.kinds.append("bytes")
        self._save_index()
        return entry_id

    # --------------------------------------------------------------- read
    def get(self, entry_id: int, auto_heal: bool = True):
        """Return the EXACT payload of an entry (read-through heal).

        ndarray entries come back as float64 (float32 fan-out, padded to
        DIM); bytes entries come back as the raw bytes."""
        if self.kinds[entry_id] == "bytes":
            return self.get_bytes(entry_id, auto_heal=auto_heal)
        v = self._vault(entry_id)
        try:
            payload = v.extract_bytes()
        except VaultError:
            if not auto_heal:
                raise
            v.heal()
            payload = v.extract_bytes()
        return unpack_value(payload)

    def get_bytes(self, entry_id: int, auto_heal: bool = True) -> bytes:
        """Return the raw payload bytes of an entry (bit-exact)."""
        v = self._vault(entry_id)
        try:
            return v.extract_bytes()
        except VaultError:
            if not auto_heal:
                raise
            v.heal()
            return v.extract_bytes()

    def lookup(self, query, tick=None, top: int = 5, gate: bool = True) -> dict:
        """Steiner-attention lookup; returns weights, active hexad, and top ids."""
        if self.n_entries == 0:
            raise VaultError("empty archive: nothing to look up yet")
        if len(self.mem.values) == 0:
            raise VaultError(
                f"archive has {self.n_entries} entries but no routable address "
                "rows in the memory layer; keys were rejected by the hive "
                "insertion heuristic"
            )
        r = self.mem.lookup(query, tick=tick, gate=gate)
        w = r["weights"]
        order = np.argsort(-w)
        topw = [(int(i), float(w[i])) for i in order[:top] if w[i] > 0]
        return {
            "weights": w,
            "active": r["active"],
            "hexad": int(r["hexad"]),
            "top": topw,
            "aggregate": r["value"],
        }

    # -------------------------------------------------------------- damage
    def kill_atoms(self, atoms) -> list:
        """Simulate atom death: zero the semantic slices AND delete the
        corresponding shard files in every entry vault."""
        atoms = sorted(int(a) for a in atoms if 0 <= int(a) < N_ATOMS)
        for i in range(self.n_entries):
            v = self._vault(i)
            for a in atoms:
                f = v._shard_file(a)
                if f.exists():
                    f.unlink()
        self.mem.kill(atoms)
        return atoms

    # ----------------------------------------------------------------- heal
    def heal(self) -> dict:
        """Heal both planes: AR-HAK semantic (approximate) + exact shard regen."""
        mem_summary = self.mem.heal()
        shards_healed: list = []
        entries_healed = 0
        for i in range(self.n_entries):
            v = self._vault(i)
            need = v.heal_report()  # raises if fewer than k survive
            if need:
                shards_healed.extend(v.heal())
                entries_healed += 1
        ver = self.verify()
        return {
            "mem": mem_summary,
            "shards_healed_total": len(shards_healed),
            "entries_healed": entries_healed,
            "verified": ver,
        }

    # -------------------------------------------------------------- verify
    def verify(self) -> dict:
        shards_ok = 0
        shards_total = 0
        entries_clean = 0
        for i in range(self.n_entries):
            v = self._vault(i)
            vv = v.verify()
            shards_ok += vv["ok"]
            shards_total += vv["n"]
            if vv["ok"] == N_ATOMS and vv["file_ok"]:
                entries_clean += 1
        return {
            "entries_clean": entries_clean,
            "entries_total": self.n_entries,
            "shards_ok": shards_ok,
            "shards_total": shards_total,
            "all_clean": entries_clean == self.n_entries,
        }

    # ----------------------------------------------------------- persistence
    def _save_index(self) -> None:
        from ..store._io import atomic_write_text

        atomic_write_text(
            self.root / "index.json",
            json.dumps(
                {"n_entries": self.n_entries, "kinds": self.kinds}, indent=2
            ),
        )

    def save(self) -> Path:
        self._save_index()
        self.mem.save(str(self.root / "memory.npz"))
        return self.root

    def load(self) -> "AssociativeArchive":
        self.mem.load(str(self.root / "memory.npz"))
        self._load_index()
        return self