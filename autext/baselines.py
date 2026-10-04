"""Reference schedulers for a full CSS code.

- :func:`staggered_schedule` -- the all-X-then-all-Z circuit, depth ``Delta_X + Delta_Z``
- :func:`exact_schedule`     -- the full CP-SAT optimum over all edges

These operate on the *whole* code, with no structural assumption. They are the
two ends of the comparison the paper draws against: the trivial upper bound of
Eq. (3), and the (expensive) true optimum.
"""
from __future__ import annotations

from .constraints import bipartite_edge_color
from .core import CSSCode, TannerGraph
from .schedule import Schedule

__all__ = ["staggered_schedule", "exact_schedule"]


def staggered_schedule(code: CSSCode) -> Schedule:
    """All X-checks, then all Z-checks: depth ``Delta_X + Delta_Z``.

    The trivial proper baseline -- no X-CX ever follows a Z-CX on a shared
    qubit -- against which the interleaving speedup is measured.
    """
    graph = TannerGraph.build(code)
    edges = graph.edges
    EX = [e for e, (_ck, _q, ty) in enumerate(edges) if ty == "X"]
    EZ = [e for e, (_ck, _q, ty) in enumerate(edges) if ty == "Z"]
    HX, HZ = code.HX, code.HZ
    DX = max(int(HX.sum(1).max(initial=0)), int(HX.sum(0).max(initial=0)))
    DZ = max(int(HZ.sum(1).max(initial=0)), int(HZ.sum(0).max(initial=0)))
    cx = bipartite_edge_color([edges[e] for e in EX], DX)
    cz = bipartite_edge_color([edges[e] for e in EZ], DZ)
    times = [0] * len(edges)
    for k, e in enumerate(EX):
        times[e] = cx[k]
    for k, e in enumerate(EZ):
        times[e] = DX + cz[k]
    meta = dict(method="staggered", delta=code.delta, T=DX + DZ, DX=DX, DZ=DZ)
    return Schedule(code=code, graph=graph, times=times, T=DX + DZ, meta=meta)


def exact_schedule(code: CSSCode, time_limit: float = 30.0) -> Schedule | None:
    """Full CP-SAT optimum over *all* edges (no structural assumption).

    Minimises makespan subject to Eq. (1) (AllDifferent per check and per qubit)
    and Eq. (2) (even-parity properness). ``meta['optimal']`` flags a proven
    optimum. Requires ``ortools``.
    """
    from ortools.sat.python import cp_model

    graph = TannerGraph.build(code)
    edges, eidx = graph.edges, graph.eidx
    by_check: dict = {}
    by_qubit: dict = {}
    for e, (c, q, _t) in enumerate(edges):
        by_check.setdefault(c, []).append(e)
        by_qubit.setdefault(q, []).append(e)
    UB = len(edges)
    m = cp_model.CpModel()
    T = [m.NewIntVar(0, UB, f"t{e}") for e in range(len(edges))]
    for grp in list(by_check.values()) + list(by_qubit.values()):
        m.AddAllDifferent([T[e] for e in grp])
    for (a, b), shared in graph.overlaps.items():
        bvars = []
        for q in shared:
            ex = eidx[(("X", a), q)]
            ez = eidx[(("Z", b), q)]
            lt = m.NewBoolVar(f"lt_{a}_{b}_{q}")
            m.Add(T[ex] < T[ez]).OnlyEnforceIf(lt)
            m.Add(T[ex] > T[ez]).OnlyEnforceIf(lt.Not())
            bvars.append(lt)
        kk = m.NewIntVar(0, len(bvars) // 2, f"k_{a}_{b}")
        m.Add(sum(bvars) == 2 * kk)
    ms = m.NewIntVar(0, UB, "ms")
    m.AddMaxEquality(ms, T)
    m.Minimize(ms)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = 8
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    times = [int(s.Value(T[e])) for e in range(len(edges))]
    depth = int(s.Value(ms)) + 1
    meta = dict(method="exact", delta=code.delta, T=depth,
                optimal=(st == cp_model.OPTIMAL))
    return Schedule(code=code, graph=graph, times=times, T=depth, meta=meta)
