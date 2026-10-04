"""Bivariate bicycle (BB) codes -- the LP codes with a 1x1 protograph.

A BB code is ``LP(A, B)`` with ``A, B`` in ``F_2[G]`` for the abelian group
``G = Z_l x Z_m``, i.e. a two-block group-algebra code. Writing
``x = S_l (x) I_m`` and ``y = I_l (x) S_m`` for the two commuting cyclic shifts,

    H_X = [A | B],    H_Z = [B^T | A^T],

with ``A``, ``B`` sums of monomials ``x^i y^j``. Commutation is automatic:
every block of ``H_X H_Z^T`` is ``AB + BA``, and ``A``, ``B`` are polynomials in
the commuting ``x, y``, so it vanishes over ``F_2``.

These are the codes of Bravyi et al. (arXiv:2308.07915) and the benchmark
family of Zhang et al. (arXiv:2603.21499) and Strikis, Browne and Beverland
(arXiv:2603.05481). ``CATALOGUE`` holds the seven instances of Bravyi et al.'s
Table 3; the parameters there are *claims*, and :func:`check_catalogue`
recomputes ``n`` and ``k`` from the matrices rather than trusting them.
"""
from __future__ import annotations

import numpy as np

from .core import CSSCode

__all__ = ["bb_code", "CATALOGUE", "gross_code", "check_catalogue"]


def _shift(n: int) -> np.ndarray:
    """The cyclic shift ``S_n``: ``S[i, j] = 1`` iff ``j == i + 1 (mod n)``."""
    return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)


def _poly(l: int, m: int, terms) -> np.ndarray:
    """Sum of monomials ``x^i y^j`` over ``F_2``, each term given as ``(i, j)``."""
    x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))
    out = np.zeros((l * m, l * m), dtype=np.uint8)
    for i, j in terms:
        out ^= (np.linalg.matrix_power(x.astype(int), i)
                @ np.linalg.matrix_power(y.astype(int), j) % 2).astype(np.uint8)
    return out


def bb_code(l: int, m: int, a_terms, b_terms, name: str = "") -> CSSCode:
    """The BB code for ``A = sum x^i y^j`` over ``a_terms`` and likewise ``B``.

    ``a_terms`` and ``b_terms`` are iterables of exponent pairs ``(i, j)``.
    """
    A, B = _poly(l, m, a_terms), _poly(l, m, b_terms)
    HX = np.hstack([A, B])
    HZ = np.hstack([B.T, A.T])
    code = CSSCode(HX, HZ, name=name or f"BB({l},{m})")
    assert not (code.HX @ code.HZ.T % 2).any(), "BB code does not commute"
    return code


# Bravyi et al., Table 3.  (l, m, A exponents, B exponents, claimed
# [[n, k, d]]).
# x^i is (i, 0) and y^j is (0, j).
CATALOGUE = {
    "[[72,12,6]]":   (6,  6,  [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    "[[90,8,10]]":   (15, 3,  [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)]),
    "[[108,8,10]]":  (9,  6,  [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    "[[144,12,12]]": (12, 6,  [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    "[[288,12,18]]": (12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)]),
    "[[360,12,24]]": (30, 6,  [(9, 0), (0, 1), (0, 2)], [(0, 3), (25, 0), (26, 0)]),
    "[[756,16,34]]": (21, 18, [(3, 0), (0, 10), (0, 17)], [(0, 5), (3, 0), (19, 0)]),
}


def from_catalogue(tag: str) -> CSSCode:
    """Build a catalogue entry by its ``[[n,k,d]]`` tag."""
    l, m, a, b = CATALOGUE[tag]
    return bb_code(l, m, a, b, name=tag)


def gross_code() -> CSSCode:
    """The gross code ``[[144,12,12]]`` of Bravyi et al."""
    return from_catalogue("[[144,12,12]]")


def check_catalogue():
    """Recompute ``n`` and ``k`` for every catalogue entry; yields rows."""
    for tag, (l, m, a, b) in CATALOGUE.items():
        code = bb_code(l, m, a, b, name=tag)
        want_n, want_k, _ = [int(v) for v in tag.strip("[]").split(",")]
        yield tag, code.n, code.k, want_n, want_k, code.n == want_n and code.k == want_k


if __name__ == "__main__":
    print(f"{'tag':>16}  {'n':>4} {'k':>3}   {'claimed n':>9} {'k':>3}  ok")
    for tag, n, k, wn, wk, ok in check_catalogue():
        print(f"{tag:>16}  {n:>4} {k:>3}   {wn:>9} {wk:>3}  {'OK' if ok else 'MISMATCH'}")
