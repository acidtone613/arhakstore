"""GF(2^8) engine tests."""

import pytest

from arhakstore.store.gf256 import GF256


@pytest.fixture(scope="module")
def f():
    return GF256()


def test_primitive_generator_cycles(f):
    seen = set()
    x = 1
    for i in range(255):
        seen.add(x)
        x = f.mul(x, f.gen)
        assert x == f.alpha(i + 1)
    assert len(seen) == 255
    assert x == 1


def test_add_is_xor(f):
    assert f.add(0xA5, 0x5A) == 0xFF
    assert f.add(3, 3) == 0


def test_inv_for_all_units(f):
    for a in range(1, 256):
        assert f.mul(a, f.inv(a)) == 1


def test_mul_inv_relation(f):
    for a in range(1, 256):
        for b in range(1, 256):
            if a ^ b == 0:
                continue
            assert f.div(f.mul(a, b), b) == a


def test_pow_is_repeated_mul(f):
    assert f.pow(7, 0) == 1
    x = 1
    for _ in range(10):
        x = f.mul(x, 7)
    assert f.pow(7, 10) == x
    assert f.pow(f.gen, 255) == 1


def test_log_consistent_with_mul(f):
    a, b = 0x8E, 0xD3
    assert f.exp[(f.log[a] + f.log[b]) % 255] == f.mul(a, b)


def test_poly_divmod_remainder_of_generator_is_zero(f):
    # G(x) = prod_{i=0}^{r-1}(x - alpha^i); generator built from its roots
    rs_gen = [1]
    for i in range(8):
        rs_gen = f.poly_mul(rs_gen, [f.alpha(i), 1])
    q, r = f.poly_divmod(rs_gen, rs_gen)
    assert not any(r)


def test_poly_val_horner(f):
    # (x - alpha^2)(x - alpha^5) evaluated at alpha^2 must be 0
    poly = f.poly_mul([f.alpha(9), 1], [f.alpha(2), 1])
    assert f.poly_val(poly, f.alpha(9)) == 0
    assert f.poly_val(poly, f.alpha(2)) == 0