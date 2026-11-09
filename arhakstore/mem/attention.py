#!/usr/bin/env python3
"""
attention.py -- Steiner-sparse attention for AR-HAK.

Mirrors RC-343 (Adaptive Hexad Attention Kernel):
  - 132 hexad signatures in R^46 (RC-343:89-90)
  - H1: hexad discrimination  (route a query to its nearest hexad)
  - H3: 23-tick phase gating   (resonant hexads only; ~95.7% FLOPs cut)
  - H4: cross-hexad bridges    (attention fidelity scales with shared atoms;
        >=3 shared atoms -> >=0.93 fidelity threshold, RC-343:195-197)
"""

import numpy as np

from .designs import FIVE_TO_HEXAD, HEXAD_MOD23, HEXADS, HEXAD_SETS, overlap_matrix

SIG_DIM = 46


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class SteinerAttention:
    """Steiner-sparse attention head over the 132-hexad Witt design."""

    def __init__(self, sig_dim: int = SIG_DIM, seed: int = 343, min_shared: int = 3):
        rng = np.random.default_rng(seed)
        self.sig_dim = sig_dim
        self.min_shared = min_shared
        self.hexads = HEXADS
        self.hexad_sets = HEXAD_SETS
        self.five_to_hexad = FIVE_TO_HEXAD
        self.mod23 = HEXAD_MOD23
        self.overlap = overlap_matrix(HEXAD_SETS)
        self.hexad_sigs = normalize(rng.standard_normal((len(HEXADS), sig_dim)))

    # -- routing ---------------------------------------------------------
    def route_5subset(self, five: tuple) -> int:
        """five -> primary hexad index (RC-343 H2 composition)."""
        return self.five_to_hexad[tuple(sorted(five))]

    def route_query(self, sig: np.ndarray) -> int:
        """46D query signature -> nearest hexad (RC-343 H1 discrimination)."""
        sig = np.asarray(sig, dtype=float).reshape(-1)
        d = self.hexad_sigs @ normalize(sig)
        return int(np.argmax(d))

    # -- masks -----------------------------------------------------------
    def bridge_mask(self, hexad: int, min_shared: int | None = None) -> np.ndarray:
        """Boolean mask over hexads sharing >= min_shared atoms with `hexad`
        (RC-343 H4 cross-hexad bridge: shared atoms drive attention fidelity)."""
        t = self.min_shared if min_shared is None else min_shared
        return self.overlap[hexad] >= t

    def resonant(self, tick: int) -> np.ndarray:
        """Boolean mask over hexads resonant at tick t (RC-343 H3, H6)."""
        return (self.mod23 * tick) % 23 == 0

    def scoped(self, query: np.ndarray, tick: int | None = None,
               min_shared: int | None = None, gate: bool = True) -> np.ndarray:
        """Full 132-dim mask: nearest-hexad bridge, optionally phase-gated.
        Returns (hexad_index, mask_over_hexads)."""
        h = self.route_query(query)
        mask = self.bridge_mask(h, min_shared)
        if gate and tick is not None:
            mask = mask & self.resonant(tick)
        return h, mask

    # -- scoring ---------------------------------------------------------
    def attention(self, query: np.ndarray, entry_sigs: np.ndarray,
                  entry_hexads: np.ndarray, tick: int | None = None,
                  min_shared: int | None = None, gate: bool = True,
                  T: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
        """Steiner-sparse attention weights over KV entries.

        Returns (weights (M,), active_mask (M,) bool, query_hexad).
        entries outside the bridge mask get weight 0 (RC-343 H4 sparsity).
        """
        h, hexad_mask = self.scoped(query, tick, min_shared, gate)
        # entry lives in a bridge hexad iff its hexad is in hexad_mask
        active = hexad_mask[entry_hexads]
        scores = np.zeros(len(entry_sigs))
        sig = normalize(np.asarray(query, dtype=float).reshape(1, -1))
        if active.any():
            sims = entry_sigs[active] @ sig.T
            scores[active] = np.asarray(sims).ravel()
        # softmax over active entries only (inactive = -inf)
        safe = np.where(active, scores, -1e9)
        safe = safe - safe.max()
        weights = np.exp(safe / T)
        weights[~active] = 0.0
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights, active, h