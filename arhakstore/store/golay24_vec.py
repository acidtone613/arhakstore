"""Vectorized Golay-24 nearest-codeword decode (AR-HAK technique, store side).

The scalar store.golay24.GolayCode decoder corrects <= 3 bit errors per
24-bit unit via a syndrome table. This module ports AR-HAK's numpy
brute-force ("decode against all 4096 codewords") trick onto the storage
side: given a batch of received 24-bit units it finds the nearest codeword
by Hamming distance and returns (messages, ok, distances). Behaviorally
identical to GolayCode.decode_24 for every unit within distance t=3, but
flags out-of-range units instead of raising, so thousands of shard headers
can be batch-checked cheaply.
"""

from __future__ import annotations

import numpy as np

from .golay24 import GolayCode, K

N24 = 24


class GolayDecoder24:
    """Batch [24,12,8] decoder consistent with GolayCode.encode_24."""

    def __init__(self):
        gc = GolayCode()
        words = np.array([gc.encode_24(m) for m in range(1 << K)], dtype=np.int32)
        bits = ((words[:, None] >> np.arange(N24)) & 1).astype(np.int8)
        self._cu = bits * 2 - 1  # (4096, 24) in {+1,-1}
        # systematic code: array index == encoded message value

    def decode(self, units) -> tuple:
        """Decode a batch of 24-bit units.

        Returns (messages int64 (M,), ok bool (M,), distances int64 (M,)).
        `ok` is False for units farther than t=3 from every codeword.
        """
        u = np.asarray(units, dtype=np.int64)
        if u.ndim == 0:
            u = u[None]
        bits = ((u[:, None] >> np.arange(N24)) & 1).astype(np.int8) * 2 - 1
        # dot of {+1,-1} length-24 vectors == 24 - 2*hamming_distance
        dists = (N24 - (bits @ self._cu.T)) // 2
        best = np.argmin(dists, axis=1)
        d = dists[np.arange(len(u)), best]
        return best, (d <= 3).astype(bool), d


def decode_batch(units, decoder: GolayDecoder24 | None = None) -> tuple:
    """Module-level convenience: messages, ok, distances for a unit batch."""
    if decoder is None:
        decoder = GolayDecoder24()
    return decoder.decode(units)