"""GF(2) linear algebra utilities.

Everything in the project ultimately reduces to row operations over
:math:`\\mathbb F_2`: ranks (for the logical-qubit count ``k``), null spaces
(for CSS logicals and the contamination polarity subspace), and reduced row
echelon form underneath both.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rref_gf2", "nullspace_gf2", "rank_gf2"]


def rref_gf2(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2).

    Returns ``(R, pivot_cols)`` where ``R`` is the RREF of ``A`` and
    ``pivot_cols`` lists the pivot column indices in order.
    """
    R = (np.asarray(A) % 2).astype(np.uint8).copy()
    m, n = R.shape
    pivot_cols: list[int] = []
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if R[i, c]), None)
        if piv is None:
            continue
        R[[r, piv]] = R[[piv, r]]
        for i in range(m):
            if i != r and R[i, c]:
                R[i] ^= R[r]
        pivot_cols.append(c)
        r += 1
        if r == m:
            break
    return R, pivot_cols


def rank_gf2(A: np.ndarray) -> int:
    """Rank of ``A`` over GF(2)."""
    return len(rref_gf2(A)[1])


def nullspace_gf2(A: np.ndarray) -> list[np.ndarray]:
    """Basis of ``{x : A x = 0 (mod 2)}`` as a list of length-``n`` vectors."""
    A = (np.asarray(A) % 2).astype(np.uint8)
    n = A.shape[1]
    R, pivot_cols = rref_gf2(A)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    basis: list[np.ndarray] = []
    for fc in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[fc] = 1
        for ri, pc in enumerate(pivot_cols):
            x[pc] = R[ri, fc] & 1
        basis.append(x % 2)
    return basis
