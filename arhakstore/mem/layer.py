#!/usr/bin/env python3
"""
layer.py -- MemoryLayer: an LLM memory layer with self-healing shards.

Data model (mirrors RC-353 holographic atom shards):
  values     (M, n_atoms*block)   memory content sharded across 48 atoms.
  codes      (M, 48)              [48,12,16] codewords on the address plane.
  sigs       (M, 46)              normalized content signatures (Steiner keys).
  hexads     (M,)                 Witt 5->hexad routing slot per entry.
  dead       set[int]             atoms whose slices are currently lost.

Flow:
  write()    project key -> 12-bit address (code) + 46D content signature.
  lookup()   Steiner-sparse attention read over the bank (RC-343 A-HAK).
  kill()     simulate atom death (zero its value slices).
  heal()     3-level repair: ECC -> Z7 entropy restoral -> re-encode.
"""

import numpy as np

from .attention import SteinerAttention, normalize
from .cores import encode
from .designs import atom_to_z7
from .repair import Repair


def five_from_bits(bits: np.ndarray) -> tuple:
    """Deterministic 5-subset of the 12 address atoms from a 12-bit
    address, guaranteeing a valid Witt routing slot for every address.

    Sorting key: set bits first (then by index) -- a stable, total order
    over the 12-bit address space, so every address maps to a hexad.
    """
    key = bits.astype(int) * 100 - np.arange(12)
    order = np.argsort(-key)
    five = tuple(sorted(int(a) for a in order[:5]))
    return five


class MemoryLayer:
    def __init__(self, block: int = 1, n_atoms: int = 48, sig_dim: int = 46,
                 seed: int = 343, min_shared: int = 3, max_dist: int = 7):
        self.block = block
        self.n_atoms = n_atoms
        self.sig_dim = sig_dim
        self.min_shared = min_shared
        self.max_dist = max_dist
        self.seed = seed
        rng = np.random.default_rng(seed)
        # fixed content projector: arbitrary key dim -> sig_dim signatures
        self.proj = rng.standard_normal((sig_dim, n_atoms * block)) / (n_atoms * block) ** 0.5
        # fixed address projector: arbitrary key dim -> 12 address bits
        self.addr_proj = rng.standard_normal((12, n_atoms * block))
        self.attention = SteinerAttention(sig_dim=sig_dim, seed=seed,
                                          min_shared=min_shared)
        self.repair = Repair(n_atoms=n_atoms, block=block, max_dist=max_dist)
        self.z7 = atom_to_z7(n_atoms)
        self.reset()

    def reset(self):
        self.values = np.zeros((0, self.n_atoms * self.block), dtype=float)
        self.codes = np.zeros((0, 48), dtype=np.uint8)
        self.sigs = np.zeros((0, self.sig_dim), dtype=float)
        self.hexads = np.zeros(0, dtype=int)
        self.meta = []
        self.dead = set()

    # -- helpers -----------------------------------------------------------
    def _address_bits(self, kv: np.ndarray) -> np.ndarray:
        slots = self.n_atoms * self.block
        m = np.zeros(slots)
        m[:min(kv.size, slots)] = kv[:min(kv.size, slots)]
        return (self.addr_proj @ m > 0).astype(np.uint8)

    def _content_sig(self, kv: np.ndarray) -> np.ndarray:
        slots = self.n_atoms * self.block
        m = np.zeros(slots)
        m[:min(kv.size, slots)] = kv[:min(kv.size, slots)]
        return normalize((self.proj @ m).reshape(1, -1)).ravel()

    # -- write -------------------------------------------------------------
    def write(self, key_vec: np.ndarray, value_vec: np.ndarray,
              meta: object = None, bits: np.ndarray | None = None) -> int:
        kv = np.asarray(key_vec, dtype=float).reshape(-1)
        vv = np.asarray(value_vec, dtype=float).reshape(-1)
        assert vv.size == self.n_atoms * self.block, \
            f"value must be {self.n_atoms*self.block}D (got {vv.size})"

        if bits is None:
            bits = self._address_bits(kv)
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        assert bits.size == 12, "address bits must be 12"

        sig = self._content_sig(kv)
        hexad = self.attention.route_query(sig)   # content-based, like lookup

        self.values = np.vstack([self.values, vv[None, :]])
        self.codes = np.vstack([self.codes, encode(bits[None, :])])
        self.sigs = np.vstack([self.sigs, sig[None, :]])
        self.hexads = np.concatenate([self.hexads, [hexad]])
        self.meta.append(meta)

        row = self.values.shape[0] - 1
        if self.dead:                       # never expose dead slices as live
            self._zero_dead_slices([row])
        return row

    # -- read --------------------------------------------------------------
    def lookup(self, query: np.ndarray, tick: int | None = None,
               min_shared: int | None = None, gate: bool = True) -> dict:
        assert len(self.values) > 0, "empty memory"
        q = np.asarray(query, dtype=float).reshape(-1)
        q_sig = self._content_sig(q)
        weights, active, h = self.attention.attention(
            q_sig, self.sigs, self.hexads, tick=tick,
            min_shared=min_shared, gate=gate)
        out = np.zeros(self.n_atoms * self.block)
        if weights.sum() > 0:
            out = weights @ self.values
        return {'value': out, 'weights': weights, 'active': active,
                'hexad': h, 'n_active': int(active.sum())}

    # -- heal ---------------------------------------------------------------
    def kill(self, atoms) -> None:
        atoms = [int(a) for a in np.atleast_1d(atoms)]
        self.dead |= set(a for a in atoms if a < self.n_atoms)
        if len(self.values):
            self._zero_dead_slices()

    def heal(self) -> dict:
        if not self.dead:
            return {'dead_atoms': 0, 'entries_healed': 0,
                    'exact_recovered': 0, 'atoms_healed': 0}
        values_h, codes_h, recovered, valid, summary = self.repair.heal(
            self.values, self.codes, self.dead)
        self.values = values_h
        self.codes = codes_h
        summary['entries_healed'] = int(len(self.values))
        summary['atoms_healed'] = len(self.dead)
        self.dead = set()
        return summary

    def _zero_dead_slices(self, rows: list[int] | None = None):
        rows = rows if rows is not None else list(range(len(self.values)))
        if not rows or not self.dead:
            return
        for a in self.dead:
            sl = slice(a * self.block, (a + 1) * self.block)
            self.values[np.array(rows), sl] = 0.0

    # -- persistence ----------------------------------------------------------
    def save(self, path: str):
        np.savez_compressed(path, values=self.values, codes=self.codes,
                            sigs=self.sigs, hexads=self.hexads,
                            dead=np.array(sorted(self.dead)),
                            meta=np.array(self.meta, dtype=object),
                            block=self.block, n_atoms=self.n_atoms)
        return path

    def load(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.values, self.codes = z['values'], z['codes']
        self.sigs, self.hexads = z['sigs'], z['hexads']
        self.dead = set(int(a) for a in z['dead'])
        self.meta = list(z['meta'])
        return self

    # -- introspection --------------------------------------------------------
    def stats(self) -> dict:
        return {
            'entries': len(self.values),
            'atoms': self.n_atoms,
            'dead': sorted(self.dead),
            'live_capacity': self.n_atoms - len(self.dead),
            'waste_ratio': len(self.dead) / self.n_atoms if self.n_atoms else 0.0,
        }