"""CSS code distance: probabilistic estimate (random-window) and exact CP-SAT.

The distance is ``d = min(dX, dZ)`` where ``dX`` is the minimum weight of an
X-type logical -- a vector ``v`` in ``ker(H_Z)`` that is not in ``rowspace(H_X)``
-- and ``dZ`` is the dual. :func:`probabilistic_distance` finds low-weight
logicals by random column-window reduction (fast, an upper bound that is tight
for LDPC codes); :func:`exact_distance` proves the minimum with CP-SAT.
"""
from __future__ import annotations

import numpy as np

from .linalg import nullspace_gf2, rank_gf2, rref_gf2

__all__ = ["probabilistic_distance", "exact_distance"]


def _rowspace(H):
    """Nonzero RREF rows of ``H`` and their pivot columns (a reduced basis)."""
    R, piv = rref_gf2(H)
    return R[:len(piv)], piv


def _in_rowspace(rows, piv, v) -> bool:
    v = (v % 2).astype(np.uint8).copy()
    for r, c in zip(rows, piv):
        if v[c]:
            v ^= r
    return not v.any()


def _logical_basis(H_stab, H_dual_checks, k):
    """A basis of the logicals: ``ker(H_dual_checks)`` modulo ``rowspace(H_stab)``."""
    M = H_stab.copy()
    rk = rank_gf2(M)
    basis = []
    for v in nullspace_gf2(H_dual_checks):
        stacked = np.vstack([M, v]) if M.size else v.reshape(1, -1)
        if rank_gf2(stacked) > rk:
            basis.append(v.astype(np.uint8))
            M, rk = stacked, rk + 1
        if len(basis) >= k:
            break
    return basis


# --------------------------------------------------------------------------
# probabilistic (random-window) distance
# --------------------------------------------------------------------------
def _prob_min_weight(H_check, H_stab, trials, rng):
    """Min weight of a vector in ``ker(H_check)`` not in ``rowspace(H_stab)``."""
    n = H_check.shape[1]
    ker = nullspace_gf2(H_check)
    if not ker:
        return None
    K = np.array(ker, dtype=np.uint8)
    stab_rows, stab_piv = _rowspace(H_stab)
    best = n + 1
    for _ in range(trials):
        perm = rng.permutation(n)
        R, piv = rref_gf2(K[:, perm])
        inv = np.argsort(perm)
        for row in R[:len(piv)][:, inv]:
            w = int(row.sum())
            if 0 < w < best and not _in_rowspace(stab_rows, stab_piv, row):
                best = w
    return best if best <= n else None


def probabilistic_distance(code, trials: int = 2000, seed: int = 0):
    """Estimate ``(d, dX, dZ)`` by random-window reduction (upper bound on ``d``)."""
    rng = np.random.default_rng(seed)
    dx = _prob_min_weight(code.HZ, code.HX, trials, rng)   # X-logicals: ker(HZ) \ rowspace(HX)
    dz = _prob_min_weight(code.HX, code.HZ, trials, rng)   # Z-logicals: ker(HX) \ rowspace(HZ)
    finite = [d for d in (dx, dz) if d is not None]
    return (min(finite) if finite else None), dx, dz


# --------------------------------------------------------------------------
# exact CP-SAT distance
# --------------------------------------------------------------------------
def _cpsat_min_weight(H_check, H_stab, logicals, time_limit):
    """Exact min weight of a nontrivial logical (anticommutes with some ``logicals``)."""
    from ortools.sat.python import cp_model

    n = H_check.shape[1]
    m = cp_model.CpModel()
    v = [m.NewBoolVar(f"v{i}") for i in range(n)]
    # v in ker(H_check): each check has even overlap with v
    for row in H_check:
        supp = [i for i in np.nonzero(row)[0]]
        if supp:
            a = m.NewIntVar(0, len(supp) // 2, "")
            m.Add(sum(v[i] for i in supp) == 2 * a)
    # nontrivial: anticommutes with at least one dual logical
    parities = []
    for lz in logicals:
        supp = [i for i in np.nonzero(lz)[0]]
        pbit = m.NewBoolVar("")
        a = m.NewIntVar(0, len(supp) // 2, "")
        m.Add(sum(v[i] for i in supp) == 2 * a + pbit)
        parities.append(pbit)
    m.Add(sum(parities) >= 1)
    m.Minimize(sum(v))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = 8
    st = s.Solve(m)
    if st == cp_model.OPTIMAL:
        return int(s.ObjectiveValue()), True
    if st == cp_model.FEASIBLE:
        return int(s.ObjectiveValue()), False   # upper bound only
    return None, False


def exact_distance(code, time_limit: float = 60.0):
    """Exact ``(d, dX, dZ)`` via CP-SAT; each value is a dict-free int.

    Each direction returns the proven optimum if solved, else the best upper
    bound found within ``time_limit`` (flagged via the return of
    :func:`_cpsat_min_weight`). Requires ``ortools``.
    """
    k = code.k
    z_logicals = _logical_basis(code.HZ, code.HX, k)   # Z-logicals for X nontriviality
    x_logicals = _logical_basis(code.HX, code.HZ, k)   # X-logicals for Z nontriviality
    dx, dx_opt = _cpsat_min_weight(code.HZ, code.HX, z_logicals, time_limit)
    dz, dz_opt = _cpsat_min_weight(code.HX, code.HZ, x_logicals, time_limit)
    d = min(d for d in (dx, dz) if d is not None) if (dx or dz) else None
    return d, dx, dz, (dx_opt and dz_opt)
