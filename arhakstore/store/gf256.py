"""GF(2^8) polynomial-basis arithmetic for the Reed-Solomon profiles used here.

The field matches the engine RS(253, 221, 33) over GF(2^8).
Dependency-free reference implementation; every symbol is one byte.

Field polynomial: 0x11D = x^8 + x^4 + x^3 + x^2 + 1.
"""

from __future__ import annotations

_POLY = 0x11D


def _gf_mul_poly(a: int, b: int, poly: int) -> int:
    """Carry-less multiply of two 8-bit values mod a degree-8 polynomial."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= poly
    return r & 0xFF


def _find_primitive(poly: int) -> int:
    """Smallest generator g whose powers cycle through all 255 units."""
    for g in range(2, 256):
        cur = 1
        seen = set()
        for _ in range(255):
            cur = _gf_mul_poly(cur, g, poly)
            seen.add(cur)
            if cur == 1:
                break
        if len(seen) == 255 and cur == 1:
            return g
    raise ValueError(f"no primitive element mod 0x{poly:X}")


def _build_tables(poly: int, gen: int):
    exp = [0] * 512
    log = [0] * 256
    cur = 1
    for i in range(255):
        exp[i] = cur
        log[cur] = i
        cur = _gf_mul_poly(cur, gen, poly)
    if cur != 1:
        raise ValueError(f"generator {gen} not primitive mod 0x{poly:X}")
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


class GF256:
    """A GF(2^8) field with precomputed exp/log lookup tables."""

    def __init__(self, poly: int = _POLY):
        self.poly = poly
        self.gen = _find_primitive(poly)
        self.exp, self.log = _build_tables(poly, self.gen)

    # -- arithmetic ----------------------------------------------------------
    def add(self, a: int, b: int) -> int:
        return a ^ b

    def sub(self, a: int, b: int) -> int:
        return a ^ b  # char 2

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % 255]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^8)")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % 255]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("inverse of zero in GF(2^8)")
        return self.exp[(255 - self.log[a]) % 255]

    def pow(self, a: int, e: int) -> int:
        if e == 0:
            return 1
        if a == 0:
            return 0
        return self.exp[(self.log[a] * e) % 255]

    def alpha(self, k: int) -> int:
        """Field element alpha^k (alpha = generator of the multiplicative group)."""
        return self.exp[k % 255]

    # -- polynomial helpers (coefficients low-degree-first) ------------------
    def poly_mul(self, a: list, b: list) -> list:
        out = [0] * (len(a) + len(b) - 1)
        for i, av in enumerate(a):
            if av == 0:
                continue
            la = self.log[av]
            for j, bv in enumerate(b):
                if bv:
                    out[i + j] ^= self.exp[(la + self.log[bv]) % 255]
        return out

    def poly_divmod(self, num: list, den: list):
        """Return (quotient, remainder) for polynomial long division.

        num/den are coefficient lists, lowest degree first.
        den must be monic of degree >= 1.
        """
        num = list(num)
        den = list(den)
        if not den or den[-1] != 1:
            raise ValueError("divisor must be monic with nonzero leading term")
        if len(num) < len(den):
            return [0], num
        q = [0] * (len(num) - len(den) + 1)  # quotient may be unused
        while len(num) >= len(den):
            shift = len(num) - len(den)
            top = num[-1]
            if not top:
                num.pop()
                continue
            coef = top  # den monic => quotient coefficient is the leading term
            q[shift] = coef
            for j, dv in enumerate(den):
                num[shift + j] ^= self.exp[(self.log[dv] + self.log[coef]) % 255] \
                    if dv else 0
            # drop the cleared leading coefficient
            num.pop()
            while len(num) > 1 and num[-1] == 0:
                num.pop()
        return q, num

    def poly_val(self, coeff: list, x: int) -> int:
        """Horner evaluation of the polynomial at field element x."""
        v = 0
        for c in reversed(coeff):
            v = self.mul(v, x) ^ c
        return v