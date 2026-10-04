"""The staged pipeline: rank cheaply, certify rarely.

This follows the driver of Strikis, Browne and Beverland (arXiv:2603.05481)
(``lr_circuits/main.py``), with our bound types on the end. Their order, and
the reason for it:

    1  residual-distance dictionary   BP-OSD once per DISTINCT residual
    2  rank every candidate           table lookup, essentially free
    3  BP-OSD on the shortlist        d_ext estimate + stim upper bound
    4  certify the winner             exact d_ext (lower), stim (upper), bracket

The per-candidate cost is dominated by the exact solver, so the pipeline is
arranged never to reach it for more than a handful of circuits. Stages 1--3
score *upper* bounds and residual proxies; those rank and reject but never
certify. Only stage 4 produces :class:`.bounds.Bound` objects on ``d_circ``,
and only a lower bound meeting an upper bound closes a :class:`.bounds.Bracket`.

Runs that certify via CP-SAT on the DEM belong in :mod:`.bench`, not here.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .bounds import bracket, lower_ext, min_over_bases, upper_bposd, upper_stim
from .build import sector
from .residual import extended_matrices, rank, tail_matrix, tails

__all__ = ["screen", "pipeline", "stim_parallel"]


def _ext_estimate(code, times, graph, basis, attempts):
    sec = sector(code, basis)
    R = tail_matrix(code, tails(code, times, graph, basis))
    He, Le = extended_matrices(sec.H, sec.L, R)
    return upper_bposd(He, Le, attempts=attempts, basis=basis, quantity="d_ext")


def screen(code, graph, candidates, *, est_attempts: int = 12, stim_lim: int = 4,
           p: float = 1e-3, verbose: bool = True):
    """Cheap per-candidate scores: ``{label: (d_ext_est_min, stim_min)}``.

    Both are min over the two bases. Both are UPPER-ish estimates (BP-OSD's
    ``d_ext`` estimate can overstate ``d_ext``; stim is a true upper bound on
    ``d_circ``), so they rank and reject but never certify.
    """
    t0 = time.time()
    out = {}
    for i, (lab, times) in enumerate(candidates):
        de, st = [], []
        for b in ("Z", "X"):
            e = _ext_estimate(code, times, graph, b, est_attempts)
            u = upper_stim(code, times, graph, b, lim=stim_lim, p=p)
            de.append(e.value if e else None)
            st.append(u.value if u else None)
        dmin = min([v for v in de if v is not None], default=None)
        smin = min([v for v in st if v is not None], default=None)
        out[lab] = (dmin, smin)
        if verbose and (i + 1) % 10 == 0:
            print(f"    screened {i + 1}/{len(candidates)} ({time.time() - t0:.0f}s)",
                  flush=True)
    return out


def pipeline(code, graph, candidates, *, top: int = 5, rank_attempts: int = 4,
             est_attempts: int = 12, stim_lim: int = 4, ext_time: float = 400.0,
             p: float = 1e-3, verbose: bool = True):
    """Rank ``candidates`` (``[(label, times), ...]``), certify the winner.

    Order, cheapest first, each stage seeing every candidate the previous one
    kept:

        1  screen        BP-OSD d_ext estimate + stim upper bound, both bases,
                         on EVERY candidate
        2  rank          by (d_ext estimate, stim) on the worse basis; the
                         residual-profile key breaks ties only
        3  certify       exact d_ext (lower) + stim (upper) on the winner

    The residual key is a tie-breaker, not the primary rank: it is often
    constant across candidates that the ``d_ext`` estimate separates. Returns a
    dict with ``screen``, ``ranked``, ``winner``, ``per_basis`` and ``total``
    (the winner's min-over-bases Bracket).
    """
    t0 = time.time()
    sc = screen(code, graph, candidates, est_attempts=est_attempts,
                stim_lim=stim_lim, p=p, verbose=verbose)
    rk, _dicts = rank(code, candidates, graph, attempts=rank_attempts, verbose=False)
    tie = {lab: key for key, lab in rk}

    def order(lab):
        d, st = sc[lab]
        return (d if d is not None else -1, st if st is not None else -1, tie[lab])

    ranked = sorted(sc, key=order, reverse=True)
    if verbose:
        print(f"    screened + ranked {len(candidates)} candidates "
              f"({time.time() - t0:.0f}s); top {top}: "
              f"{[(str(l), sc[l]) for l in ranked[:top]]}", flush=True)
    by_label = dict(candidates)
    winner = ranked[0]
    times = by_label[winner]
    per = {}
    for b in ("Z", "X"):
        lo = lower_ext(code, times, graph, b, time_limit=ext_time)
        up = upper_stim(code, times, graph, b, lim=stim_lim, p=p)
        per[b] = bracket(lo, up, basis=b)
        if verbose:
            print(f"    winner {b}: {lo}  |  {up}  ->  {per[b]}", flush=True)
    total = min_over_bases(per["Z"], per["X"])
    if verbose:
        print(f"    winner {winner}: {total}  ({time.time() - t0:.0f}s total)",
              flush=True)
    return {"screen": sc, "ranked": ranked, "winner": winner, "per_basis": per,
            "total": total}


# ------------------------------------------------------------ parallel stim
_CTX: dict = {}


def _init(HX, HZ):
    from ..core import CSSCode, TannerGraph

    code = CSSCode(HX, HZ)
    _CTX["code"], _CTX["graph"] = code, TannerGraph.build(code)


def _one(args):
    times, basis, lim, p, deg_lim, no_increase = args
    b = upper_stim(_CTX["code"], np.asarray(times), _CTX["graph"], basis, lim=lim, p=p,
                   deg_lim=deg_lim, no_increase=no_increase)
    return None if b is None else b.value


def stim_parallel(code, graph, candidates, lim: int = 4, p: float = 1e-3,
                  workers=None, deg_lim=None, no_increase: bool = False):
    """stim upper bound (min over bases) for every candidate, across processes.

    The caller's script MUST be behind ``if __name__ == "__main__":`` --- Windows
    spawns workers by re-importing the main module, and without the guard each
    worker re-runs the driver and spawns more workers.
    """
    workers = workers or min(os.cpu_count() or 4, 12)
    tasks, index = [], []
    for lab, times in candidates:
        for basis in ("Z", "X"):
            tasks.append((np.asarray(times).tolist(), basis, lim, p, deg_lim, no_increase))
            index.append(lab)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(code.HX, code.HZ)) as ex:
        vals = list(ex.map(_one, tasks, chunksize=1))
    out: dict = {}
    for lab, v in zip(index, vals):
        if v is not None:
            out[lab] = v if lab not in out else min(out[lab], v)
    for lab, _t in candidates:
        out.setdefault(lab, None)
    return out
