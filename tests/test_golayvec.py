"""Vectorized Golay-24 decoder vs the scalar table decoder (consistency)."""

import random

import numpy as np
import pytest

from arhakstore.store.golay24 import GolayCode
from arhakstore.store.golay24_vec import GolayDecoder24


@pytest.fixture(scope="module")
def golay():
    return GolayCode()


@pytest.fixture(scope="module")
def vec():
    return GolayDecoder24()


def test_vec_matches_scalar_under_3_errors(golay, vec):
    rng = random.Random(0x2A)
    units = []
    expect_msg = []
    for _ in range(400):
        msg = rng.randrange(4096)
        unit = golay.encode_24(msg)
        n_err = rng.randint(1, 3)
        for p in rng.sample(range(24), n_err):
            unit ^= 1 << p
        units.append(unit)
        expect_msg.append(msg)
    msgs, ok, dists = vec.decode(np.array(units, dtype=np.int64))
    assert ok.all()
    assert (msgs == np.array(expect_msg)).all()


def test_vec_flags_uncorrectable(golay, vec):
    # 4+ errors are never .ok => caller treats them as corrupt, not silent.
    bad = [golay.encode_24(0x1AB) ^ ((1 << 0) | (1 << 5) | (1 << 11) | (1 << 22))]
    msgs, ok, dists = vec.decode(np.array(bad))
    assert not ok[0]
    assert dists[0] > 3


def test_vec_scalar_agree_on_batch_vault_headers(golay, vec):
    rng = random.Random(7)
    indices = [rng.randrange(4096) for _ in range(200)]
    units = [golay.encode_24(i) for i in indices]
    msgs, ok, _ = vec.decode(np.array(units, dtype=np.int64))
    assert ok.all()
    assert (msgs == np.array(indices)).all()  # index == message (systematic)