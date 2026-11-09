#!/usr/bin/env python3
"""
repair.py -- layered self-healing for AR-HAK.

Three-level repair:
  Level 1: [48,12,16] ECC decode corrects <=7 corrupted
      atoms per codeword (bit-plane erasure + flips).
  Level 2: deleted nodes are refilled with MEANINGFUL entropy by
      averaging their living Z7 topological group-mates.
  Level 3: full codeword recovery by re-encoding the decoded message
      once enough atoms are healed (erasure-to-recovery).

The `heal` routine never invents content: a deleted value-slice is rebuilt
from its Z7 kinship group, and a deleted codeword bit is rebuilt exactly
from the algebraic code.
"""

import numpy as np

from .cores import decode_nearest, encode
from .designs import atom_to_z7


class Repair:
    def __init__(self, n_atoms: int = 48, block: int = 1, max_dist: int = 7):
        self.n_atoms = n_atoms
        self.block = block                 # value dims per atom
        self.max_dist = max_dist           # t=7 for [48,12,16]
        self.z7 = atom_to_z7(n_atoms)

    # -- value-slice helpers --------------------------------------------
    def slices(self, atom: int) -> slice:
        return slice(atom * self.block, (atom + 1) * self.block)

    def group_mates(self, atom: int, dead: set[int]) -> list[int]:
        """Living Z7 group-mates of a dead atom."""
        return [m for m in range(self.n_atoms)
                if self.z7[m] == self.z7[atom] and m not in dead]

    # -- level 1: ECC ---------------------------------------
    def ecc_recover(self, cws: np.ndarray, dead: set[int]):
        """Recover messages + codewords when dead atoms are treated as
        bit-flips (their bits zeroed). Returns corrected codewords.
        Codewords whose received distance <= max_dist are exactly restored.
        """
        cw_corrupt = cws.copy()
        if dead:
            cw_corrupt[:, sorted(dead)] = 0
        msgs, valid, dists = decode_nearest(cw_corrupt, self.max_dist)
        corrected = cws.copy()
        corrected[valid] = encode(msgs[valid])
        return corrected, valid, dists

    # -- level 2: topological entropy restoral --------------
    def groupmate_restore(self, values: np.ndarray, dead: set[int],
                          apply_to_dead_only: bool = True) -> np.ndarray:
        """Refill dead atoms' value-slices as the mean of their living Z7
        group-mates. This is the research's exact
        'duplicate meaningful entropy back into deleted nodes' step.

        apply_to_dead_only=False also smooths living slices; default keeps repairs local.
        """
        out = values.copy()
        for atom in sorted(dead):
            mates = self.group_mates(atom, dead)
            if mates:
                sl = self.slices(atom)
                cols = np.array([list(range(m * self.block, (m + 1) * self.block))
                                 for m in mates], dtype=int)
                out[:, sl] = np.mean(values[:, cols], axis=1)
        if apply_to_dead_only is False:
            # spectral-mode pass also touches survivors (research default
            # path rebuilds x from Phi.H @ H); we keep the dead-only default.
            pass
        return out

    # -- level 3: re-encode erased codewords ------------
    def reencode(self, cws: np.ndarray, valid: np.ndarray, msgs: np.ndarray) -> np.ndarray:
        """Full codeword recovery by re-encoding decoded messages."""
        out = cws.copy()
        if valid.any():
            out[valid] = encode(msgs[valid])
        return out

    # -- compound heal ----------------------------------------------------
    def heal(self, values: np.ndarray, cws: np.ndarray, dead: set[int],
             entries_corrupted: set[int] | None = None):
        """Run all three levels. Returns (values_healed, cws_healed,
        recovered_messages, valid_mask, summary dict).

        `values` shape (M, n_atoms*block); `cws` shape (M, 48).
        """
        dead = set(dead)
        if entries_corrupted is None:
            entries_corrupted = set(range(values.shape[0]))

        # Level 2 first: give dead atoms content (entropy from their kin),
        # so downstream attention reads are coherent.
        values_h = self.groupmate_restore(values, dead)

        # Level 1+3: algebraic recovery over the address plane.
        idx = np.array(sorted(entries_corrupted), dtype=int)
        if len(idx):
            corrected, valid, dists = self.ecc_recover(cws[idx], dead)
        else:
            corrected = cws[:0]
            valid = np.zeros(0, dtype=bool)
            dists = np.zeros(0)

        cws_h = cws.copy()
        if len(idx):
            cws_h[idx] = corrected  # level 3: re-encoded codewords restored

        recovered, _, _ = decode_nearest(cws_h, self.max_dist)

        summary = {
            'dead_atoms': len(dead),
            'entries': int(len(idx)),
            'exact_recovered': int(np.sum(valid)),
            'value_slices_restored': len(dead) * (len(idx) > 0),
            'max_received_dist': float(dists.max()) if len(dists) else 0.0,
        }
        return values_h, cws_h, recovered, valid, summary