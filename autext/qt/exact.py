"""Exact depth-T feasibility (CP-SAT), to separate `hard obstruction' from
`greedy weakness'.

The model is the scheduling problem itself, nothing more: one integer variable
z_e per edge, AllDifferent at every square, AllDifferent at every check, and
max(group i) < min(group i+1) at every square.  Variable domains are shrunk to
the window `core.windows` (a consequence of those constraints, not an extra
one), which is a search speedup only: the verdict does not depend on it.

Speaks `core.Inst`, and every schedule it returns is refereed by `core.verify`.
"""
import sys
import numpy as np
from ortools.sat.python import cp_model
from . import core
from .build import from_qt
from .qt import QT


def solve(inst, T, tl=120, workers=8, seed=None, forbid_tails=None, forbid_combos=None):
    """Is there a depth-T schedule of `inst`?  Returns (status, sched) with
    status in FEASIBLE / INFEASIBLE / INFEASIBLE(degree) / INFEASIBLE(window) /
    unknown, and `sched` an int array of 1-based layers when FEASIBLE.

    `seed` randomises the search so repeated calls on the SAME instance return
    DIFFERENT depth-T schedules.  Without it CP-SAT is deterministic and every
    instance yields a single schedule, which starves any search over the
    schedule space.

    `forbid_tails` maps a check id to a list of edge-index sets that may NOT be
    a SUFFIX of that check's CNOT order.  A suffix is exactly a residual (hook)
    error, so forbidding the low-residual-distance ones steers the scheduler
    away from circuits that can fail cheaply; this is the idea of Strikis,
    Browne and Beverland (arXiv:2603.05481) applied as a CONSTRAINT rather than
    as a filter after the fact.  `S` is a suffix iff
    min(z over S) > max(z over the rest); layers within a check are all
    different, so forbidding it is just the strict reverse."""
    if inst.Delta() > T:
        return 'INFEASIBLE(degree)', None
    lo, hi = core.windows(inst, T)
    if np.any(lo > hi):
        return 'INFEASIBLE(window)', None

    m = cp_model.CpModel()
    z = [m.NewIntVar(int(lo[e]), int(hi[e]), '') for e in range(inst.m)]
    bysq, byck = {}, {}
    for e in range(inst.m):
        bysq.setdefault(int(inst.sq[e]), {}).setdefault(int(inst.grp[e]), []).append(e)
        byck.setdefault(int(inst.ck[e]), []).append(e)
    # `forbid_combos`: a list of combinations, each a list of (check id, edge
    # set).  Every combination must have AT LEAST ONE member that is NOT a
    # suffix of its check.  Each is a no-good learnt from an extended-code
    # witness: the residuals that together realised the minimum d_ext.
    if forbid_combos:
        for combo in forbid_combos:
            bs = []
            for c, S in combo:
                es = byck.get(int(c), [])
                inS = [e for e in es if e in S]
                rest = [e for e in es if e not in S]
                if not inS or not rest:
                    continue
                mS = m.NewIntVar(1, T, '')
                MR = m.NewIntVar(1, T, '')
                m.AddMinEquality(mS, [z[e] for e in inS])
                m.AddMaxEquality(MR, [z[e] for e in rest])
                b = m.NewBoolVar('')
                m.Add(mS < MR).OnlyEnforceIf(b)          # b: S is NOT a suffix
                m.Add(mS > MR).OnlyEnforceIf(b.Not())
                bs.append(b)
            if bs:
                m.AddBoolOr(bs)

    if forbid_tails:
        for c, bad in forbid_tails.items():
            es = byck.get(int(c), [])
            for S in bad:
                inS = [e for e in es if e in S]
                rest = [e for e in es if e not in S]
                if not inS or not rest:
                    continue
                mS = m.NewIntVar(1, T, '')
                MR = m.NewIntVar(1, T, '')
                m.AddMinEquality(mS, [z[e] for e in inS])
                m.AddMaxEquality(MR, [z[e] for e in rest])
                m.Add(mS < MR)          # S is not a suffix

    for v, gs in bysq.items():
        m.AddAllDifferent([z[e] for g in gs for e in gs[g]])
        prev_max = None
        for g in sorted(gs):
            mn = m.NewIntVar(1, T, ''); mx = m.NewIntVar(1, T, '')
            m.AddMinEquality(mn, [z[e] for e in gs[g]])
            m.AddMaxEquality(mx, [z[e] for e in gs[g]])
            if prev_max is not None:
                m.Add(prev_max < mn)
            prev_max = mx
    for c, es in byck.items():
        m.AddAllDifferent([z[e] for e in es])

    s = cp_model.CpSolver()
    if seed is not None:
        s.parameters.random_seed = int(seed)
        s.parameters.randomize_search = True
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_search_workers = workers
    st = s.Solve(m)
    name = {cp_model.OPTIMAL: 'FEASIBLE', cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE', cp_model.UNKNOWN: 'unknown'}[st]
    sched = None
    if name == 'FEASIBLE':
        sched = np.fromiter((s.Value(z[e]) for e in range(inst.m)),
                            np.int32, inst.m)
        core.verify(inst, sched, T)      # never report an unverified schedule
    return name, sched


if __name__ == '__main__':
    PAIRS = [('[2,1,2]', '[7,4,3]'), ('[4,2,2]', '[7,4,3]'),   # layer-Hall FAILS
             ('[6,3,3]', '[6,3,3]'), ('[7,4,3]', '[5,2,3]'),   # layer-Hall ok,
             ('[7,4,3]', '[6,3,3]'), ('[8,4,4]', '[5,2,3]'),   # greedy misses
             ('[7,4,3]', '[8,4,4]')]
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    print('=== exact feasibility at T = Delta, systematic halving ===')
    for a, b in PAIRS:
        qt = QT(a, b)
        k, kp = qt.nA // 2, qt.nB // 2
        rho = np.array([1] * (qt.nA - k) + [0] * k)
        sigma = np.array([0] * kp + [1] * (qt.nB - kp))
        inst = from_qt(qt, rho, sigma)
        T = qt.Delta()
        r, _ = solve(inst, T, tl=tl)
        print(f'  {a+" x "+b:>22s}  Delta={T:3d}  |E|={inst.m:5d}  ->  {r}')
