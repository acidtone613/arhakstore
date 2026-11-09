"""Reed-Solomon codes over GF(2^8), systematic form.

Implements three RS profiles over GF(2^8):

  Engine     RS(253, 221, 33)  k=221  t=16  erasure budget 32
  Quadallic  RS(48, 12, 37)    k=12   t=18  erasure budget 36
  Binary     RS(23, 12, 12)    k=12   t=5   erasure budget 11

Design notes (honest limits):
  * Encoding is systematic: the codeword's last k symbols are the message.
  * Two decode paths are provided:
      - ERASURE reconstruction (`repair`/`repairer`): positions known, e.g.
        whole shard files lost or hash-flagged corrupt. Any k of n survive.
      - ERROR + ERASURE correction (`decode_correct`): Berlekamp-Massey +
        Forney. Corrects any pattern with 2*nu + mu <= n - k random symbol
        errors (nu) and declared erasures (mu). This is the full production
        decoder.
  * The generator polynomial roots are alpha^0..alpha^(r-1) (fcr = 0), so a
    length-r syndrome vector is r(alpha^i).
  * Generator matrices are cached per (n, k); inverting the k x k survivor
    submatrix is the only O(k^3) step and happens once per restore/heal.
"""

from __future__ import annotations

from .gf256 import GF256

ENGINE = "engine"
QUADALLIC = "quadallic"
BINARY = "binary"

PROFILES = {
    ENGINE: (253, 221, "Engine RS(253,221,33)  n=253 k=221 t=16  ~14% overhead"),
    QUADALLIC: (48, 12, "Quadallic RS(48,12,37)  n=48 k=12 t=18  deep archive"),
    BINARY: (23, 12, "Binary RS(23,12,12)  n=23 k=12 t=5  toy/demo"),
}


class DecodeError(ValueError):
    """Decoding went beyond the code's guaranteed correction radius."""


def _poly_scale(f: GF256, p: list, x: int) -> list:
    return [f.mul(c, x) for c in p]


def _poly_add(f: GF256, a: list, b: list) -> list:
    out = a[:]
    if len(b) > len(out):
        out += [0] * (len(b) - len(out))
    for i, bv in enumerate(b):
        out[i] ^= bv
    return out


def _poly_trim(p: list) -> list:
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _poly_derivative(f: GF256, p: list) -> list:
    # In characteristic 2 only odd-degree coefficients survive.
    out = []
    for k in range(1, len(p)):
        if k & 1:  # coefficient k * p[k]; k mod 2 is 1 for odd k
            out.append(p[k])
        else:
            out.append(0)
    return _poly_trim(out)


_GENERATOR_CACHE: dict = {}


class ReedSolomon:
    """A systematic (n, k) Reed-Solomon code over GF(2^8)."""

    def __init__(self, n: int, k: int):
        if not (1 <= k < n <= 255):
            raise ValueError(f"need 1 <= k < n <= 255, got n={n}, k={k}")
        self.n = n
        self.k = k
        self.r = n - k
        self.field = GF256()
        self._g = None
        self._gen = self._generator()

    # ------------------------------------------------------------------ setup
    def _generator(self) -> list:
        """G(x) = prod_{i=0}^{r-1} (x - alpha^i), monic, as coeff list."""
        f = self.field
        g = [1]
        for i in range(self.r):
            g = f.poly_mul(g, [f.alpha(i), 1])  # (x - a^i) == (x + a^i) in char 2
        return g

    def generator_matrix(self) -> list:
        """k x n matrix whose row i is the codeword for message e_i.

        c_j = sum_i m_i * G[i][j]  (message m as a row vector).

        Cached per (n, k): the generator depends only on the code, never on
        the survivor set, so it is built once per code not once per vault.
        """
        if self._g is None:
            global _GENERATOR_CACHE
            key = (self.n, self.k)
            g = _GENERATOR_CACHE.get(key)
            if g is None:
                g = []
                for i in range(self.k):
                    row = [0] * self.k
                    row[i] = 1
                    g.append(list(self.encode(bytes(row))))
                _GENERATOR_CACHE[key] = g
            self._g = g
        return self._g

    # -------------------------------------------------------------- encoding
    def encode(self, msg: bytes | bytearray) -> bytes:
        """Systematic encode: k message symbols -> n codeword symbols.

        Layout: c(x) = m(x) * x^r + (m(x) * x^r mod G(x)), so in the returned
        byte string the message occupies the LAST k symbols and the parity
        occupies the first r = n-k symbols (increasing polynomial degree).
        """
        msg = bytes(msg)
        if len(msg) != self.k:
            raise ValueError(f"message must be {self.k} symbols, got {len(msg)}")
        f = self.field
        r = self.r
        # m(x) * x^r  (low-degree-first coefficient list)
        _, rem = f.poly_divmod([0] * r + list(msg), self._gen)
        rem = (rem + [0] * r)[:r]
        return bytes(rem) + msg

    # ----------------------------------------------------------- syndromes
    def syndromes(self, received: list) -> list:
        """S_i = r(alpha^i) for i in 0..r-1 (generator roots, fcr = 0)."""
        f = self.field
        return [f.poly_val(received, f.alpha(i)) for i in range(self.r)]

    # ---------------------------------------------- errors + erasures decoder
    def decode_correct(self, received: bytes | list, erasures: list | None = None) -> bytes:
        """Correct any pattern with 2*nu + mu <= n - k against `received`.

        `received` is a full length-n symbol vector; `erasures` lists symbol
        positions treated as known-unknown (0-placeholder allowed). Returns
        the corrected length-n codeword. Raises DecodeError when the received
        word lies beyond the decoding radius (including miscorrection that
        fails the locator checks).
        """
        rx = list(bytes(received))
        if len(rx) != self.n:
            raise ValueError(f"received must be {self.n} symbols, got {len(rx)}")
        f = self.field
        erases = [int(p) for p in (erasures or [])]
        if any(p < 0 or p >= self.n for p in erases):
            raise DecodeError(f"erasure position out of range: {erases}")
        erases = sorted(set(erases))
        if len(erases) > self.r:
            raise DecodeError(
                f"too many erasures: {len(erases)} (budget r = {self.r})"
            )
        # The tight capability bound 2*nu + mu <= r is enforced by the
        # Berlekamp-Massey locator check below (nu is discovered there).

        synd = self.syndromes(rx)
        if all(s == 0 for s in synd):
            return bytes(rx)  # zero syndromes: codeword already valid

        g = self._erasures_locator(erases)
        fsynd = (f.poly_mul(synd, g))[: self.r]  # S(x) * Gamma(x) mod x^r
        err_loc = self._find_error_locator(fsynd, len(erases))
        errata_loc = _poly_trim(f.poly_mul(g, err_loc))
        oo = _poly_trim(f.poly_mul(synd, errata_loc))[: self.r]

        errata_positions = self._errata_positions(errata_loc)
        deg = len(errata_loc) - 1
        if len(errata_positions) != deg:
            raise DecodeError(
                f"found {len(errata_positions)} errata roots, expected {deg}; "
                "received word is beyond the code's correction radius"
            )

        deriv = _poly_derivative(f, errata_loc)
        out = rx[:]
        for p in errata_positions:
            X = f.alpha(p % 255)  # position element alpha^p
            x = f.inv(X)  # x_k = X_k^-1 = alpha^-p (a locator root)
            mag_num = f.mul(X, f.poly_val(oo, x))
            mag_den = f.poly_val(deriv, x)
            if mag_den == 0:
                raise DecodeError("singular errata locator derivative")
            out[p] ^= f.div(mag_num, mag_den)
        return bytes(out)

    def _erasures_locator(self, erases: list) -> list:
        f = self.field
        loc = [1]
        for pos in erases:
            loc = f.poly_mul(loc, [1, f.alpha(pos % 255)])
        return loc

    def _find_error_locator(self, forney_synd: list, erase_count: int) -> list:
        """Berlekamp-Massey on the (erasure-folded) syndrome sequence.

        Classic formulation with explicit bounds: the discrepancy at step n
        reads the syndrome at offset n + erase_count, and terms with negative
        index are zero (polynomial coefficients below degree 0 do not exist).
        Returns the error locator Lambda (constant coefficient 1).
        """
        f = self.field
        nsym = self.r
        seq = forney_synd  # full length-nsym sequence (Forney syndromes)
        r2 = nsym - erase_count
        size = r2 + 1
        C = [1] + [0] * size
        B = [1] + [0] * size
        L = 0
        m = 1
        b = 1
        for n in range(r2):
            idx = n + erase_count
            d = seq[idx]
            for i in range(1, L + 1):
                ci = C[i]
                if ci and idx - i >= 0:
                    d ^= f.mul(ci, seq[idx - i])
            if d == 0:
                m += 1
            elif 2 * L <= n:
                T = C[:]
                coef = f.mul(d, f.inv(b))
                for i in range(0, size - m):
                    bi = B[i]
                    if bi:
                        C[i + m] ^= f.mul(coef, bi)
                L = n + 1 - L
                B, b, m = T, d, 1
            else:
                coef = f.mul(d, f.inv(b))
                for i in range(0, size - m):
                    bi = B[i]
                    if bi:
                        C[i + m] ^= f.mul(coef, bi)
                m += 1
        err_loc = _poly_trim(C[: L + 1])
        errs = max(0, len(err_loc) - 1)
        if 2 * errs + erase_count > nsym:
            raise DecodeError(
                f"too many errors to correct: locator degree {errs} exceeds "
                f"budget (2*{errs} + {erase_count} > {nsym} syndromes)"
            )
        return err_loc

    def _errata_positions(self, errata_loc: list) -> list:
        f = self.field
        positions = []
        for p in range(self.n):
            x = f.alpha((255 - (p % 255)) % 255)  # alpha^-p
            if f.poly_val(errata_loc, x) == 0:
                positions.append(p)
        return positions

    def correct_message(self, received: bytes | list, erasures: list | None = None) -> bytes:
        """Decode and return the k-symbol systematic message."""
        return self.decode_correct(received, erasures)[self.r:]

    # -------------------------------------------------------------- erasure
    @staticmethod
    def _gauss_invert(mat: list, f: GF256) -> list:
        """Invert a k x k matrix over GF(2^8) via Gauss-Jordan with pivoting."""
        n = len(mat)
        a = [row[:] for row in mat]
        inv = [[0] * n for _ in range(n)]
        for i in range(n):
            inv[i][i] = 1
        for col in range(n):
            pivot = next((row for row in range(col, n) if a[row][col] != 0), None)
            if pivot is None:
                raise ValueError("singular generator submatrix (data-dependent loss)")
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
                inv[col], inv[pivot] = inv[pivot], inv[col]
            d = f.inv(a[col][col])
            if d != 1:
                a[col] = [f.mul(x, d) for x in a[col]]
                inv[col] = [f.mul(x, d) for x in inv[col]]
            for row in range(n):
                if row == col:
                    continue
                coef = a[row][col]
                if coef:
                    for j in range(n):
                        a[row][j] ^= f.mul(a[col][j], coef)
                        inv[row][j] ^= f.mul(inv[col][j], coef)
        return inv

    def repairer(self, positions: list) -> "ReedSolomonRepairer":
        """Precompute a recoverer for a fixed survivor-position set (O(k^3) once).

        Use this instead of repair() in loops: reconstruction afterwards is
        O(k^2 + k*n) per codeword instead of O(k^3).
        """
        positions = list(positions)
        if len(positions) < self.k:
            raise ValueError(f"need at least {self.k} survivors, got {len(positions)}")
        return ReedSolomonRepairer(self, positions[: self.k])

    def repair(self, survivors: dict) -> bytes:
        """Rebuild a full length-n codeword from k surviving (pos, symbol) pairs."""
        return self.repairer(list(survivors)).recover(survivors)


class ReedSolomonRepairer:
    """Repeated full-codeword reconstruction for a fixed survivor set.

    `recover_symbols(survivors, want)` precomputes the k-vector transfer
    coefficients per wanted position once, so healing M dead symbols of C
    chunks costs M * O(k^2) + M*C * O(k) after a single k x k inversion.
    """

    def __init__(self, rs: ReedSolomon, positions: list):
        self.rs = rs
        self.positions = list(positions)[: rs.k]
        f = rs.field
        g = rs.generator_matrix()
        # Solve cs = m * M where M[i][j] = G[i][positions[j]] (k x k).
        self._M = [[g[i][p] for p in self.positions] for i in range(rs.k)]
        self._inv = ReedSolomon._gauss_invert(self._M, f)
        self._g = g
        self._T = {}  # position j -> k coefficients (symbol_j = sum_p cs_p T[p][j])

    def recover(self, survivors: dict) -> bytes:
        rs, f = self.rs, self.rs.field
        cs = [survivors[p] for p in self.positions]
        msg = self._message(cs)
        log_m = [0] * rs.k
        for i, mi in enumerate(msg):
            if mi:
                log_m[i] = f.log[mi]
        # precompute per-column (row, log coeff) lists once
        rows_by_col = self._rows_by_col()
        codeword = bytearray(rs.n)
        for j in range(rs.n):
            acc = 0
            for i, log_g in rows_by_col[j]:
                if msg[i]:
                    acc ^= f.exp[(log_m[i] + log_g) % 255]
            codeword[j] = acc
        return bytes(codeword)

    def _message(self, cs: list) -> list:
        rs, f = self.rs, self.rs.field
        inv = self._inv
        msg = [0] * rs.k
        for i in range(rs.k):
            acc = 0
            for j in range(rs.k):
                acc ^= f.mul(cs[j], inv[j][i])
            msg[i] = acc
        return msg

    def _rows_by_col(self):
        rows = getattr(self, "_rows", None)
        if rows is None:
            f = self.rs.field
            rows = []
            for j in range(self.rs.n):
                cols = []
                for i in range(self.rs.k):
                    gi = self._g[i][j]
                    if gi:
                        cols.append((i, f.log[gi]))
                rows.append(cols)
            self._rows = rows
        return rows

    def _transfer(self, positions: list) -> list:
        """k-vector T_j per wanted position: symbol_j = sum_p cs_p * T_j[p]."""
        rs, f = self.rs, self.rs.field
        inv, g = self._inv, self._g
        todo = [j for j in positions if j not in self._T]
        if todo:
            K = rs.k
            for j in todo:
                col = [0] * K
                for i in range(K):
                    gi = g[i][j]
                    if gi:
                        for p in range(K):
                            ip = inv[p][i]
                            if ip:
                                col[p] ^= f.mul(ip, gi)
                self._T[j] = col
        return [self._T[j] for j in positions]

    def recover_symbols(self, survivors: dict, want: list) -> list:
        """Rebuild only the requested codeword positions (exact, fast)."""
        rs, f = self.rs, self.rs.field
        cs = [survivors[p] for p in self.positions]
        transfers = self._transfer(list(want))
        out = []
        for j, col in zip(want, transfers):
            acc = 0
            for p in range(rs.k):
                cp = cs[p]
                cpw = col[p]
                if cp and cpw:
                    acc ^= f.exp[(f.log[cp] + f.log[cpw]) % 255]
            out.append(acc)
        return out