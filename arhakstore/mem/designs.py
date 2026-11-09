#!/usr/bin/env python3
"""
designs.py -- Steiner geometry for AR-HAK.

- 132 hexads over 12 atoms with guaranteed 5-subset coverage (RC-343:35-86).
- Z7 topological atom map (RC-353:83):  atom_to_z7 = ((s*17+31) % 48) % 7
- 23-tick phase wheel for resonance gating (RC-343:150-154).
"""

import itertools
import os

import numpy as np

N_HEXADS = 132
N_ATOMS = 12
N_PHASES = 23
N_ATOMS_48 = 48

_HEXAD_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "witt_hexads.npy")


def build_hexads() -> tuple[list[tuple], list[set], dict[tuple, int]]:
    """Return (hexads, hexad_sets, five_to_hexad) with full 5-subset coverage.

    Uses the true small Witt design S(5,6,12): the 132 hexads are the orbit
    of one block under PSL(2,11) acting on P^1(F_11). Every one of the 792
    five-subsets is covered exactly once, giving the total RC-343 routing
    table (any 5-atom query signature maps to a hexad via `five_to_hexad`).
    """
    if not os.path.exists(_HEXAD_DATA):
        raise RuntimeError(f"witt_hexads.npy missing at {_HEXAD_DATA}")
    arr = np.load(_HEXAD_DATA)
    hexads = [tuple(int(x) for x in row) for row in arr]
    assert len(hexads) == N_HEXADS

    # Phase-resonant slots (i % 23 == 0, RC-343 gating target) should hold a
    # maximal-union 6-subset so gated reads cover all 12 atoms evenly.
    target_slots = [i for i in range(N_HEXADS) if i % N_PHASES == 0]
    hexads = _place_resonant(hexads, target_slots)

    hexad_sets = [set(h) for h in hexads]
    five_to_hexad = {}
    for i, h in enumerate(hexads):
        for c5 in itertools.combinations(h, 5):
            five_to_hexad[tuple(sorted(c5))] = i
    assert len(five_to_hexad) == 792, f"5-subset coverage failed: {len(five_to_hexad)}/792"
    return hexads, hexad_sets, five_to_hexad


def _place_resonant(hexads: list[tuple], target_slots: list[int]) -> list[tuple]:
    """Permute a balanced 6-subset into the resonant slots.

    Phase gating (RC-343 H3) activates only the hexads at i % 23 == 0, so
    those hexads must give every atom equal representation (otherwise gated
    reads skew toward over-covered atoms and fidelity drops). We greedily
    pick the 6 hexads whose atom-occurrence counts are most uniform, then
    swap them onto the resonant slots (a pure permutation, since S(5,6,12)
    is a Steiner system and overwriting would lose 5-subset coverage).
    """
    n = len(hexads)
    coverage = np.zeros(12, dtype=float)
    chosen = []
    pool = list(range(n))
    # preferred: each of the 12 atoms appears exactly len/2 = 3 times
    target = np.full(12, 3.0)
    for _ in range(len(target_slots)):
        best, best_score = None, None  # best_score = -penalty (higher better)
        for i in pool:
            cand = coverage.copy()
            hx = set(hexads[i])
            cand[list(hx)] += 1.0
            score = -float(np.mean((cand - target) ** 2))
            if best_score is None or score > best_score:
                best, best_score = i, score
        chosen.append(best)
        pool.remove(best)
        coverage[list(set(hexads[best]))] += 1.0

    out = list(hexads)
    for slot in target_slots:
        if slot not in chosen:
            idx = chosen.pop()
            out[slot], out[idx] = out[idx], out[slot]
    return [tuple(int(x) for x in h) for h in out]


HEXADS, HEXAD_SETS, FIVE_TO_HEXAD = build_hexads()


def atom_to_z7(n_atoms: int = N_ATOMS_48) -> np.ndarray:
    """Z7 topological group per atom (RC-353:83)."""
    return np.array([((s * 17 + 31) % 48) % 7 for s in range(n_atoms)])


def z7_group_mates(atom: int, n_atoms: int = N_ATOMS_48) -> list[int]:
    """Living-group mates for a single atom under Z7 kinship."""
    z = atom_to_z7(n_atoms)
    return [m for m in range(n_atoms) if z[m] == z[atom]]


def hexad_mod23(n_hexads: int = N_HEXADS) -> np.ndarray:
    """Phase class of each hexad (RC-343:150)."""
    return np.array([i % N_PHASES for i in range(n_hexads)])


HEXAD_MOD23 = hexad_mod23()


def phase_gate(tick: int, include: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of resonant hexads at tick t (RC-343:152-154).

    A hexad i is active at tick t when (i mod 23) * t mod 23 == 0.
    """
    ids = np.arange(N_HEXADS) if include is None else np.asarray(include)
    return (HEXAD_MOD23[ids] * tick) % N_PHASES == 0


def phase_gate_ids(tick: int) -> np.ndarray:
    """Indices of resonant hexads at tick t."""
    return np.nonzero(phase_gate(tick))[0]


def overlap_matrix(hexad_sets: list[set] = None) -> np.ndarray:
    """(132,132) shared-atom intersection counts (RC-343:267-270)."""
    sets = hexad_sets if hexad_sets is not None else HEXAD_SETS
    n = len(sets)
    overlap = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            overlap[i, j] = len(sets[i] & sets[j])
    return overlap


if __name__ == "__main__":
    z = atom_to_z7()
    print(f"hexads: {len(HEXADS)}, 5-subsets covered: {len(FIVE_TO_HEXAD)}/792")
    print(f"Z7 distribution over 48 atoms: {np.bincount(z, minlength=7)}")
    print(f"resonant hexads tick=0: {len(phase_gate_ids(0))}, tick=5: {len(phase_gate_ids(5))}")