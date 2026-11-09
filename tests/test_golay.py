"""Golay [24,12,8] / [23,12,7] tests (DMF face-pair 1 cyclic construction)."""

import random

import pytest

from arhakstore.store.golay24 import GolayCode, GEN, K, MAX_ERRORS, N23


@pytest.fixture(scope="module")
def golay():
    return GolayCode()


def test_all_4096_codewords_have_minimum_distance_7(golay):
    weights = set()
    for msg in range(1, 1 << K):  # the zero codeword has weight 0 by definition
        word = golay.encode(msg)
        w = bin(word).count("1")
        weights.add(w)
        assert w >= 7, f"distance violation for msg {msg}"
    # perfect [23,12,7] weight enumerator
    assert weights == {7, 8, 11, 12, 15, 16, 23}


def test_encode_24_parity(golay):
    for msg in (0, 1, 0xFFF, 0xABC):
        unit = golay.encode_24(msg)
        c23 = unit & ((1 << 23) - 1)
        assert ((unit >> 23) & 1) == (bin(c23).count("1") & 1)


@pytest.mark.parametrize("errors", [1, 2, 3])
def test_decode_corrects_up_to_3(golay, errors):
    rng = random.Random(errors)
    for _ in range(60):
        msg = rng.randrange(1 << K)
        word = golay.encode(msg)
        positions = rng.sample(range(N23), errors)
        for p in positions:
            word ^= 1 << p
        got, fixed = golay.decode(word)
        assert got == msg
        assert fixed == errors


def test_decode_24_corrects_against_parity(golay):
    rng = random.Random(4)
    for _ in range(40):
        msg = rng.randrange(1 << K)
        unit = golay.encode_24(msg)
        # corrupt <= 3 arbitrary bit positions within the 24-bit unit
        n_err = rng.randint(1, 3)
        positions = rng.sample(range(24), n_err)
        for p in positions:
            unit ^= 1 << p
        got, fixed = golay.decode_24(unit)
        assert got == msg
        assert fixed == n_err


def test_decode_24_isolated_parity_bit(golay):
    msg = 0x123
    unit = golay.encode_24(msg)
    unit ^= 1 << 23  # corrupt the parity bit alone
    got, fixed = golay.decode_24(unit)
    assert got == msg
    assert fixed == 1


def test_four_errors_beyond_bound(golay):
    # A bounded-distance decoder corrects up to t=3; beyond that it may
    # miscorrect silently (never guaranteed), or raise. Either is legal, but
    # it must never return the ORIGINAL message with 0 errors claimed.
    msg = 0x1AB
    word = golay.encode(msg)
    for p in (0, 5, 11, 22):
        word ^= 1 << p
    try:
        got, fixed = golay.decode(word)
    except ValueError:
        return  # uncorrectable is legal
    assert not (got == msg and fixed == 0)


def test_generator_polynomial_matches_corpus():
    # g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1  == 0xC75
    assert GEN == 0xC75
    assert (N23 - K) == 11
    assert MAX_ERRORS == 3  # t = floor((7-1)/2)


def test_table_is_full_bijection(golay):
    assert len(golay._table) == (1 << (N23 - K))  # all 2048 syndromes
    # syndrome of the identity pattern is zero
    assert golay._table.get(0) == 0
    assert golay.decode(golay.encode(0)) == (0, 0)