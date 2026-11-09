"""Reed-Solomon encode / erasure-reconstruct tests over the corpus profiles."""

import os
import random

import pytest

from arhakstore.store.rs import ReedSolomon, PROFILES

PROFILES_UNDER_TEST = ["binary", "quadallic", "engine"]


@pytest.fixture(scope="module")
def rng():
    return random.Random(0xD00D)


def _rand_bytes(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


@pytest.mark.parametrize("profile", PROFILES_UNDER_TEST)
def test_roundtrip_systematic(profile):
    n, k, _ = PROFILES[profile]
    rs = ReedSolomon(n, k)
    msg = _rand_bytes(random.Random(7), k)
    code = rs.encode(msg)
    assert len(code) == n
    assert code[rs.r:] == msg  # systematic: message in the last k symbols


@pytest.mark.parametrize("profile", ["quadallic", "binary"])
def test_repair_random_erasures(profile):
    n, k, _ = PROFILES[profile]
    rs = ReedSolomon(n, k)
    rng = random.Random(99)
    for trial in range(6):
        msg = _rand_bytes(rng, k)
        code = rs.encode(msg)
        lost = rng.sample(range(n), n - k)  # destroy the maximum permissible
        survivors = {j: code[j] for j in range(n) if j not in set(lost)}
        full = rs.repair(survivors)
        assert full == code
        assert bytes(survivors[j] for j in sorted(survivors)) == bytes(
            code[j] for j in sorted(survivors)
        )


@pytest.mark.slow
def test_repair_random_erasures_engine():
    n, k, _ = PROFILES["engine"]
    rs = ReedSolomon(n, k)
    rng = random.Random(99)
    for trial in range(6):
        msg = _rand_bytes(rng, k)
        code = rs.encode(msg)
        lost = rng.sample(range(n), n - k)  # destroy the maximum permissible
        survivors = {j: code[j] for j in range(n) if j not in set(lost)}
        full = rs.repair(survivors)
        assert full == code
        assert bytes(survivors[j] for j in sorted(survivors)) == bytes(
            code[j] for j in sorted(survivors)
        )


def test_repairer_loop_matches_repair():
    rs = ReedSolomon(48, 12)
    rng = random.Random(3)
    positions = rng.sample(range(48), 12)
    repairer = rs.repairer(positions)
    for _ in range(20):
        msg = _rand_bytes(rng, 12)
        code = rs.encode(msg)
        survivors = {j: code[j] for j in positions}
        assert repairer.recover(survivors) == code
        assert rs.repair(survivors) == code


def test_repair_needs_k():
    rs = ReedSolomon(23, 12)
    with pytest.raises(ValueError):
        rs.repair({0: 1, 1: 2})


def test_engine_profile_numbers():
    n, k, _ = PROFILES["engine"]
    assert (n, k) == (253, 221)
    d = n - k + 1
    assert d == 33
    assert (d - 1) // 2 == 16  # t = 16, matching the corpus hierarchy
    assert len(PROFILES) == 3


def test_every_profile_encodes_and_reconstructs(rng):
    for profile, (n, k, _) in PROFILES.items():
        rs = ReedSolomon(n, k)
        msg = _rand_bytes(rng, k)
        code = rs.encode(msg)
        erased = set(range(0, n - k))  # drop a contiguous prefix region
        survivors = {j: code[j] for j in range(n) if j not in erased}
        assert rs.repair(survivors) == code


def test_large_message_many_chunks():
    rs = ReedSolomon(253, 221)
    rng = random.Random(1)
    chunks = [_rand_bytes(rng, 221) for _ in range(10)]
    coded = [rs.encode(c) for c in chunks]
    positions = rng.sample(range(253), 221)
    survivors = [{j: co[j] for j in positions} for co in coded]
    repairer = rs.repairer(positions)
    rebuilt = [repairer.recover(s) for s in survivors]
    assert all(a == b for a, b in zip(coded, rebuilt))
    for a in rebuilt:
        assert a[rs.r:] in chunks


# ---------------------------------------------------------------------------
# Berlekamp-Massey + Forney: errors and errors+erasures decoding
# ---------------------------------------------------------------------------

def test_decode_correct_errors_only(rng):
    """Random symbol errors (up to t) are corrected exactly, per profile."""
    for profile, (n, k, _) in PROFILES.items():
        t = (n - k) // 2
        rs = ReedSolomon(n, k)
        for trial in range(8):
            msg = _rand_bytes(rng, k)
            cw = rs.encode(msg)
            rx = bytearray(cw)
            for p in rng.sample(range(n), rng.randint(1, t)):
                rx[p] ^= rng.randrange(1, 256)
            assert rs.decode_correct(rx) == cw, (profile, trial)
            assert rs.correct_message(rx) == msg


def test_decode_correct_erasures_only(rng):
    """Whole-window erasures up to r are reconstructed (2*0 + mu <= r)."""
    for profile, (n, k, _) in PROFILES.items():
        rs = ReedSolomon(n, k)
        for trial in range(8):
            msg = _rand_bytes(rng, k)
            cw = rs.encode(msg)
            erases = rng.sample(range(n), rng.randint(1, rs.r))
            rx = bytearray(cw)
            for p in erases:
                rx[p] = 0
            assert rs.decode_correct(rx, erases) == cw, (profile, trial)


def test_decode_correct_mixed_errors_and_erasures(rng):
    """Any pattern with 2*nu + mu <= r corrects exactly."""
    for profile, (n, k, _) in PROFILES.items():
        t = (n - k) // 2
        rs = ReedSolomon(n, k)
        for trial in range(8):
            msg = _rand_bytes(rng, k)
            cw = rs.encode(msg)
            nu = rng.randint(0, min(t, 6))
            mu = rng.randint(0, rs.r - 2 * nu)
            chosen = rng.sample(range(n), nu + mu)
            erases, errs = chosen[:mu], chosen[mu:]
            rx = bytearray(cw)
            for p in erases:
                rx[p] = 0
            for p in errs:
                rx[p] ^= rng.randrange(1, 256)
            assert rs.decode_correct(rx, erases) == cw, (profile, trial, nu, mu)


def test_decode_beyond_radius_raises():
    rs = ReedSolomon(23, 12)  # t=5
    rng = random.Random(99)
    cw = rs.encode(_rand_bytes(rng, 12))
    rx = bytearray(cw)
    for p in range(6):  # t+1 errors
        rx[p] ^= rng.randrange(1, 256)
    with pytest.raises(ValueError):
        rs.decode_correct(rx)
    # ... and the vault layer must never silently accept those (Raises VaultError)


def test_decode_tolerates_valid_codeword():
    rs = ReedSolomon(48, 12)
    cw = rs.encode(_rand_bytes(random.Random(4), 12))
    assert rs.decode_correct(cw) == cw
    assert rs.decode_correct(cw, [3, 9]) == cw  # declared erases, none wrong


def test_recover_symbols_matches_recover(rng):
    rs = ReedSolomon(48, 12)
    positions = rng.sample(range(48), 12)
    repairer = rs.repairer(positions)
    for _ in range(5):
        cw = rs.encode(_rand_bytes(rng, 12))
        survivors = {j: cw[j] for j in positions}
        syms = repairer.recover_symbols(survivors, [0, 7, 33, 47])
        assert syms == [cw[j] for j in [0, 7, 33, 47]]
        assert repairer.recover(survivors) == cw


def test_transfer_memoized(rng):
    rs = ReedSolomon(23, 12)
    positions = rng.sample(range(23), 12)
    repairer = rs.repairer(positions)
    assert repairer._transfer([5]) == repairer._transfer([5])  # cached
    assert list(repairer._T) == [5]