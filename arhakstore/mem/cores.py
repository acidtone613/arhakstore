#!/usr/bin/env python3
"""
cores.py -- [48,12,16] Superbridge codec (doubled extended Golay)
Mirrors RC-340 (Superbridge Constellation) and RC-342 (Galactic Cluster).

Construction (RC-340:44-98):
    A_adj      icosahedron adjacency  (12x12)
    B_ico      J - A_adj              (Hadamard-style block)
    G_golay    [I_12 | B_ico]         -> [24,12,8] extended Golay
    G_super    [G_golay | G_golay]    -> [48,12,16] Superbridge

Decoding (RC-340:107-116): vectorized nearest-codeword over the full
4096-codeword look-up table, valid when min distance <= max_dist (t=7).
"""

import numpy as np


def build_adjacency() -> np.ndarray:
    """Icosahedron adjacency matrix A_adj (RC-340:48-58)."""
    A = np.zeros((12, 12), dtype=np.uint8)
    for i in range(5):
        A[i, (i + 1) % 5] = 1
        A[i, (i - 1) % 5] = 1
    for i in range(5):
        A[5 + i, 5 + (i + 1) % 5] = 1
        A[5 + i, 5 + (i - 1) % 5] = 1
    for i in range(5):
        A[10, i] = 1
        A[i, 10] = 1
        A[11, 5 + i] = 1
        A[5 + i, 11] = 1
    for i in range(5):
        A[i, 5 + i] = 1
        A[5 + i, i] = 1
        A[i, 5 + (i + 1) % 5] = 1
        A[5 + (i + 1) % 5, i] = 1
    return A


G_SUPER = np.hstack([
    np.hstack([np.eye(12, dtype=np.uint8), np.ones((12, 12), dtype=np.uint8) - build_adjacency()]),
    np.hstack([np.eye(12, dtype=np.uint8), np.ones((12, 12), dtype=np.uint8) - build_adjacency()]),
])


def enumerate_codewords() -> tuple[np.ndarray, np.ndarray]:
    """All 4096 codewords + messages (RC-340:67-98). Returns (messages, codewords)."""
    messages = np.zeros((4096, 12), dtype=np.uint8)
    codewords = np.zeros((4096, 48), dtype=np.uint8)
    for idx in range(4096):
        msg = np.array([(idx >> i) & 1 for i in range(12)], dtype=np.uint8)
        messages[idx] = msg
        codewords[idx] = (msg @ G_SUPER) % 2
    return messages, codewords


MESSAGES, CODEWORDS = enumerate_codewords()


def min_weight() -> int:
    w = np.sum(CODEWORDS[1:], axis=1)  # skip zero message
    return int(np.min(w[w > 0]))


def weight_counts() -> np.ndarray:
    counts = np.zeros(49, dtype=np.int64)
    for w in np.sum(CODEWORDS, axis=1):
        counts[w] += 1
    return counts


def encode(messages: np.ndarray) -> np.ndarray:
    """(...,12) message bits -> (...,48) codeword bits."""
    return (messages @ G_SUPER) % 2


def decode_nearest(received: np.ndarray, max_dist: int = 7):
    """Vectorized nearest-codeword decoding (RC-340:107-116).

    received: (...,48) codeword bits (possibly corrupted).
    Returns (decoded_messages (...,12), valid (...,) bool, dists (...,)).
    """
    flat = received.reshape(-1, 48).astype(np.uint8)
    rcvd_exp = flat[:, None, :]
    cw_exp = CODEWORDS[None, :, :]
    dists = np.sum((rcvd_exp + cw_exp) % 2, axis=2)
    min_dists = np.min(dists, axis=1)
    min_idx = np.argmin(dists, axis=1)
    decoded = MESSAGES[min_idx]
    valid = min_dists <= max_dist
    return (decoded.reshape(received.shape[:-1] + (12,)),
            valid.reshape(received.shape[:-1]),
            min_dists.reshape(received.shape[:-1]))