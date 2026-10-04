r"""Residual (hook) errors: the cheap, code-level side of Strikis, Browne and
Beverland (arXiv:2603.05481).

A fault on an ancilla partway through a check's CNOT sequence lands on the data
qubits the check has *not yet* touched. For a check with CNOT order
``i_1, ..., i_w`` the residuals are the tails ``E_l = e_{i_l} + ... + e_{i_w}``,
``l = 2..w``. Which check family's ancillas matter for a memory basis is fixed
in :func:`autext.circ.build.sector` and nowhere else.

Two quantities are computed from them, and their directions differ:

- :func:`residual_distance` --- their Def. 1,
  ``Delta(E) = 1 + min{|D| : E + D is a nontrivial logical}``, by BP-OSD.
  Thm 1 says ``d'_circ <= Delta(E)`` for EVERY residual, so ``min_E Delta(E)`` is
  an UPPER bound on ``d'_circ`` and certifies nothing about ``d_circ``. It is a
  ranking proxy, computed once per distinct residual into a dictionary and then
  looked up (their ``evaluate_residual_dists`` / ``find_CNOT_orders``).
- the extended code --- append one column per residual to ``(H, L)``; its
  exact distance is ``d'_circ`` itself. That lives in :func:`.bounds.lower_ext`.

The proxy only sees one residual at a time, while the actual minimum-weight
failure can combine several, so raising its floor need not raise ``d_circ``.
"""
from __future__ import annotations

import time

import numpy as np

from .build import sector

__all__ = ["tails", "tail_matrix", "extended_matrices", "residual_distance",
           "residual_dict", "cost_vector", "cost_key", "rank"]


def tails(code, times, graph, basis: str):
    """``{check_id: [frozenset, ...]}`` --- the residuals of every hook-producing
    check under the CNOT order ``times`` gives it, tails in order ``l = 2..w``."""
    want = sector(code, basis).hook_checks
    by_check: dict = {}
    for e, (ck, q, ty) in enumerate(graph.edges):
        if ty == want:
            by_check.setdefault(ck[1], []).append((int(times[e]), q))
    out = {}
    for cid, lst in by_check.items():
        lst.sort()
        qs = [q for _t, q in lst]
        out[cid] = [frozenset(qs[l:]) for l in range(1, len(qs))]
    return out


def tail_matrix(code, tails_by_check):
    """Distinct residuals as columns of an ``n x m`` uint8 matrix."""
    seen, cols = set(), []
    for lst in tails_by_check.values():
        for sup in lst:
            if sup in seen:
                continue
            seen.add(sup)
            v = np.zeros(code.n, dtype=np.uint8)
            v[list(sup)] = 1
            cols.append(v)
    if not cols:
        return np.zeros((code.n, 0), dtype=np.uint8)
    return np.array(cols, dtype=np.uint8).T


def extended_matrices(H, L, R):
    """``(H_ext, L_ext)``: original columns, then one per residual column of ``R``.

    A residual column costs 1 --- one fault buys the whole tail --- which is
    what makes the extended code the right model of the de-interleaved circuit.
    """
    if R.shape[1] == 0:
        return H.copy(), L.copy()
    return (np.hstack([H, (H @ R) % 2]).astype(np.uint8),
            np.hstack([L, (L @ R) % 2]).astype(np.uint8))


def residual_distance(H, L, support, attempts: int = 8, seed: int = 0,
                      error_rate: float = 0.05, osd_order: int = 6):
    """``Delta(E)`` by BP-OSD (Strikis et al. ``BPOSD_res_dist``): decode the
    residual's own syndrome with the logical bit flipped, so a solution ``D``
    has ``H(E+D) = 0`` and ``logical.(E+D) = 1``; answer ``|D| + 1``. An
    estimate: it can only overstate ``Delta``."""
    from ldpc import BpOsdDecoder

    n = H.shape[1]
    E = np.zeros(n, dtype=np.uint8)
    E[list(support)] = 1
    rng = np.random.default_rng(seed)
    best, miss = n + 1, 0
    while miss < attempts:
        v = rng.integers(0, 2, size=L.shape[0])
        while not v.any():
            v = rng.integers(0, 2, size=L.shape[0])
        row = ((v @ L) + (rng.integers(0, 2, size=H.shape[0]) @ H)) % 2
        Hs = np.vstack([H, row[None, :]]).astype(np.uint8)
        syn = (Hs @ E) % 2
        syn[-1] ^= 1
        dec = BpOsdDecoder(Hs, error_rate=error_rate, max_iter=30,
                           bp_method="ms", ms_scaling_factor=0.625,
                           osd_method="osd_cs", osd_order=osd_order)
        D = dec.decode(syn.astype(np.uint8))
        if not ((Hs @ D) % 2 != syn).any():
            wt = int(np.count_nonzero(D)) + 1
            if wt < best:
                best, miss = wt, 0
                continue
        miss += 1
    return best


def residual_dict(code, candidates, graph, basis: str, attempts: int = 4,
                  verbose: bool = False):
    """One BP-OSD call per DISTINCT residual across ALL candidates.

    Their ``evaluate_residual_dists``: built once per code, then every
    candidate is scored by lookup. Candidates share most of their residuals,
    so the union is much smaller than the sum, which is what makes ranking
    hundreds of orderings cheap.
    """
    sec = sector(code, basis)
    union = set()
    for _lab, times in candidates:
        for lst in tails(code, times, graph, basis).values():
            union.update(lst)
    out, t0 = {}, time.time()
    for i, sup in enumerate(sorted(union, key=lambda s: (len(s), sorted(s)))):
        out[sup] = residual_distance(sec.H, sec.L, sup, attempts=attempts)
        if verbose and (i + 1) % 250 == 0:
            print(f"      residual dict {i + 1}/{len(union)} ({time.time() - t0:.0f}s)",
                  flush=True)
    return out


def cost_vector(code, times, graph, basis: str, dct):
    """Histogram of residual distances by lookup: ``vec[i]`` residuals with ``Delta = i``."""
    vec = np.zeros(code.n + 2, dtype=int)
    for lst in tails(code, times, graph, basis).values():
        for sup in lst:
            vec[dct[sup]] += 1
    return vec


def cost_key(vec):
    """Strikis et al.'s ranking key, larger is better: the smallest residual
    distance present first, then fewer residuals at that level, then the
    profile."""
    nz = np.nonzero(vec)[0]
    if len(nz) == 0:
        return (10 ** 9, 0, ())
    worst = int(nz[0])
    return (worst, -int(vec[worst]), tuple(-int(v) for v in vec[worst + 1:]))


def rank(code, candidates, graph, bases=("Z", "X"), dicts=None, attempts: int = 4,
         verbose: bool = False):
    """Candidates best-first by residual profile, keyed on the WORST basis.

    ``d_circ`` is the min over bases, so a candidate is only as good as its
    weaker basis: ranking on a single basis buys distance in that basis while
    the other one lags. Returns ``([(key, label), ...], dicts)``.
    """
    dicts = dicts or {b: residual_dict(code, candidates, graph, b, attempts, verbose)
                      for b in bases}
    out = []
    for lab, times in candidates:
        key = min(cost_key(cost_vector(code, times, graph, b, dicts[b])) for b in bases)
        out.append((key, lab))
    out.sort(key=lambda r: r[0], reverse=True)
    return out, dicts
