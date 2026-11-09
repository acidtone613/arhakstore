#!/usr/bin/env python3
"""
hologram.py -- 276-slot spectral resonator.

The 23 > 276 > 48 chain concretely:
  23        tick clock / phase structure
  276       N_SLOTS of the hologram
  48        N_SPATIAL atoms of the state space

Construction is exactly (Phi_48 = F_276 @ Q with Q from the
economic QR of a random 276x48 matrix; F_276 the sqrt-normalized DFT).

Decoding: x_hat = Phi_48.H @ H with H = Phi_48 @ x, i.e.
the resonator's linear inversion is lossless while Gram(I) holds, then the
topological Z7 repair (repair.groupmate_restore) rebuilds dead atom blocks.
"""

import numpy as np


def dft_matrix(n: int) -> np.ndarray:
    """n x n sqrt-normalized unitary DFT."""
    idx = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n) / np.sqrt(n)


def build_phi48(n_slots: int = 276, n_atoms: int = 48, seed: int = 353) -> np.ndarray:
    """Phi_48 (276 x 48): F_276 @ Q over the economic QR of a random matrix. Gram -> I => perfect linear inversion."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_slots, n_atoms))
    Q, _ = np.linalg.qr(A, mode='reduced')
    F = dft_matrix(n_slots)
    return F @ Q


def gram_residual(phi: np.ndarray) -> float:
    return float(np.max(np.abs(phi.conj().T @ phi - np.eye(phi.shape[1]))))


def encode_state(phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Holographic encoding: H = Phi @ x."""
    return phi @ x


def decode_state(phi: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Linear recall: x_hat = Phi.H @ H."""
    return phi.conj().T @ H


def fidelity(x_ref: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.abs(np.vdot(x_ref, x_hat)) /
                 (np.linalg.norm(x_ref) * np.linalg.norm(x_hat) + 1e-10))


if __name__ == "__main__":
    phi = build_phi48()
    print(f"Phi_48 Gram residual: {gram_residual(phi):.2e}")
    rng = np.random.default_rng(7)
    x = rng.standard_normal(48)
    x = x / np.linalg.norm(x)
    print(f"perfect recall fidelity: {fidelity(x, decode_state(phi, encode_state(phi, x))):.6f}")