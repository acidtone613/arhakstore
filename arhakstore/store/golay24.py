"""Systematic binary Golay code (DMF face-pair 1, cyclic construction).

The corpus builds the [24,12,8] Golay code and its perfect punctured sibling
[23,12,7] from the degree-11 generator polynomial

    g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1   (= 0xC75)

(see README.md "Face Pair 1: Cyclic <-> QR"). This module implements:

  * encode(msg12)  -> 23-bit perfect-Golay codeword (systematic cyclic,
    message occupies positions 11..22, check bits 0..10)
  * decode(word23) -> (msg12, errors) correcting every pattern of <= 3 bit
    errors (t = 3), via a table over all 2048 low-weight syndromes
  * encode_24 / decode_24 -> 24-bit unit: 23-bit codeword + overall parity
    bit; a mismatched parity bit is itself corrected

The table decoder is exact: for weight <= 3 the map error -> syndrome is a
bijection onto the 2048 syndromes, so decoding is deterministic and complete.
"""

from __future__ import annotations

GEN = 0xC75          # generator polynomial binary representation
K = 12
N23 = 23
MAX_ERRORS = 3
_SYNDROME_SPACE = 1 << (N23 - K)  # 2048


def _bits_to_int(coeff_list: list) -> int:
    v = 0
    for i, b in enumerate(coeff_list):
        if b:
            v |= 1 << i
    return v


def _poly_mod_int(word: int) -> int:
    """word mod g (g = GEN), bits indexed by polynomial degree."""
    deg = N23 - K
    coeff = [(word >> i) & 1 for i in range(N23)]
    while len(coeff) - 1 >= deg:
        top = len(coeff) - 1
        if not coeff[top]:
            coeff.pop()
            continue
        for i in range(deg + 1):
            if (GEN >> i) & 1:
                coeff[top - deg + i] ^= 1
        coeff.pop()
    return _bits_to_int(coeff)


class GolayCode:
    """Systematic (24-bit / 23-bit) Golay code with table decoding."""

    def __init__(self):
        self._table = {}  # syndrome (11 bits) -> error pattern (23 bits)
        self._build_table()

    def _build_table(self):
        import itertools

        for weight in range(MAX_ERRORS + 1):
            for combo in itertools.combinations(range(N23), weight):
                pattern = 0
                for p in combo:
                    pattern |= 1 << p
                syn = _poly_mod_int(pattern)
                if syn in self._table and self._table[syn] != pattern:
                    continue  # cannot happen in the perfect code; defensive
                self._table[syn] = pattern

    # ------------------------------------------------------------------ encode
    def encode(self, msg: int) -> int:
        """Encode a 12-bit message into a 23-bit perfect-Golay codeword."""
        if not 0 <= msg < (1 << K):
            raise ValueError(f"message must be 12 bits, got {msg}")
        # c(x) = m(x) * x^11 + (m(x) * x^11 mod g); message sits at degrees 11..22
        word = msg << (N23 - K)
        num = [0] * (N23 - K) + [(msg >> i) & 1 for i in range(K)]
        rem = _poly_mod_int(sum(bit << i for i, bit in enumerate(num)))
        return word | rem

    def parity(self, word23: int) -> int:
        return bin(word23).count("1") & 1

    def encode_24(self, msg: int) -> int:
        """24-bit unit: 23-bit codeword plus overall parity bit at position 23."""
        c23 = self.encode(msg)
        p = bin(c23).count("1") & 1
        return c23 | (p << 23)

    # ------------------------------------------------------------------ decode
    def decode(self, word23: int) -> tuple:
        """Decode a 23-bit word; returns (message, number_of_errors_fixed)."""
        syn = _poly_mod_int(word23 & ((1 << N23) - 1))
        if syn == 0:
            return word23 >> (N23 - K), 0
        pattern = self._table.get(syn)
        if pattern is None:
            raise ValueError(f"uncorrectable Golay word (syndrome 0x{syn:03X}, >{MAX_ERRORS} errors)")
        corrected = word23 ^ pattern
        return corrected >> (N23 - K), bin(pattern).count("1")

    def decode_24(self, word24: int) -> tuple:
        """Decode a 24-bit unit (23-bit codeword + parity bit at position 23).

        Corrects any pattern of <= 3 bit errors across the whole 24-bit unit.
        """
        if not 0 <= word24 < (1 << 24):
            raise ValueError("word must be 24 bits")
        p = (word24 >> 23) & 1
        c23 = word24 & ((1 << 23) - 1)
        msg, err23 = self.decode(c23)  # raises if >3 errors in the 23-bit part
        codeword = self.encode(msg)
        total = err23 + (self.parity(codeword) != p)  # independent parity check
        if total > MAX_ERRORS:
            raise ValueError("uncorrectable Golay-24 unit (too many errors)")
        return msg, total