"""Witness-driven local search over depth-``Delta`` schedules (PropHunt-style).

The candidate generators in :mod:`.candidates` sample CP-SAT's solution space,
which is small and clustered. This module instead *walks*: take a valid
schedule, ask stim for the actual minimum-weight undetectable logical fault
set --- which names the CX gates involved --- and try the modifications that
could break it:

- a **Kempe chain**: pick a witness edge of layer ``t1`` and another layer
  ``t2``, follow the alternating ``t1``/``t2`` path through the Tanner graph
  (a path or a cycle, since a vertex has at most one edge of each layer) and
  swap the two layers along all of it.

Swapping just two edges' layers is NOT a move at depth ``Delta``: every check
occupies every layer and every data qubit is busy in every layer, so any
single swap collides. The Kempe chain preserves the proper colouring at every
vertex by construction; only the per-square sandwich order can then break, and
``core.verify`` is the referee for that. The objective is stim's witness
weight, min over the two memory bases, which is cheap enough that hundreds of
moves cost less than one exact solve.

This is PropHunt's loop (arXiv:2601.17580) with the circuit-distance objective:
it optimises the INTERLEAVED circuit, which the residual/``d_ext`` objectives
cannot, being provably invariant under de-interleaving. stim's witness is an
upper bound, so the search is a heuristic; the winner is certified afterwards
by the :mod:`.search` pipeline.
"""
from __future__ import annotations

import time

import numpy as np

from ..qt import core as qtcore
from .build import memory_circuit

__all__ = ["witness", "touched_edges", "score", "local_search"]


def witness(code, times0, graph, basis: str, lim: int = 4, p: float = 1e-3):
    """``(weight, faults)`` from stim, or ``(None, [])`` if nothing found.

    Each fault is ``(tick, qubits)`` with ``qubits`` the indices the flipped
    Pauli product acts on (data ``< n``, X-ancilla ``< n + rX``, else Z-ancilla).
    """
    circ = memory_circuit(code, times0, graph, basis=basis, p=p)
    try:
        errs = circ.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=lim,
            dont_explore_edges_with_degree_above=lim,
            dont_explore_edges_increasing_symptom_degree=False)
    except Exception:
        return None, []
    faults = []
    for e in errs:
        loc = e.circuit_error_locations[0]
        qs = [t.gate_target.value for t in loc.flipped_pauli_product]
        faults.append((int(loc.tick_offset), qs))
    return len(errs), faults


def touched_edges(code, graph, times0, faults):
    """Edges implicated by the faults: the CX at (check, data) for two-qubit
    faults, and every edge at the data qubit within one layer for single-qubit
    ones (an idle fault sits between two of its CXs)."""
    n, rX = code.n, code.rx
    eidx = {}
    at_q: dict = {}
    for e, (ck, q, ty) in enumerate(graph.edges):
        eidx[(ty, ck[1], q)] = e
        at_q.setdefault(q, []).append(e)
    out = set()
    for tick, qs in faults:
        data = [q for q in qs if q < n]
        anc = [q for q in qs if q >= n]
        for q in data:
            for a in anc:
                key = ("X", a - n, q) if a < n + rX else ("Z", a - n - rX, q)
                if key in eidx:
                    out.add(eidx[key])
            if not anc:
                for e in at_q.get(q, []):
                    if abs(int(times0[e]) - tick) <= 1:
                        out.add(e)
        if anc and not data:
            for a in anc:
                ck = ("X", a - n) if a < n + rX else ("Z", a - n - rX)
                for e, (c2, _q, _t) in enumerate(graph.edges):
                    if c2 == ck and int(times0[e]) >= tick:
                        out.add(e)
    return out


def score(code, times0, graph, lim: int = 4, p: float = 1e-3):
    """``(min_weight, {basis: (weight, faults)})``; ``None`` weight if unresolved."""
    per = {b: witness(code, times0, graph, b, lim=lim, p=p) for b in ("Z", "X")}
    ws = [w for w, _f in per.values() if w is not None]
    return (min(ws) if ws else None), per


def _kempe(e0, t2, times1, ends, at):
    """Edges on the alternating ``t1``/``t2`` chain through ``e0``.

    ``ends[e]`` are the two vertices of edge ``e``; ``at[(v, t)]`` is the edge
    of layer ``t`` at vertex ``v`` or ``None``. Walks both ways from ``e0``.
    """
    t1 = int(times1[e0])
    chain = {e0}
    for start_v in ends[e0]:
        v, prev, want = start_v, e0, t2
        while True:
            nxt = at.get((v, want))
            if nxt is None or nxt in chain:
                break
            chain.add(nxt)
            a, b = ends[nxt]
            v = b if a == v else a
            prev, want = nxt, (t1 if want == t2 else t2)
    return chain


def _valid(inst, times1, T):
    try:
        qtcore.verify(inst, times1, T)
        return True
    except AssertionError:
        return False


def local_search(inst, code, graph, times1, *, iters: int = 300, lim: int = 4,
                 seed: int = 0, plateau: float = 0.3, p: float = 1e-3,
                 verbose: bool = True):
    """Hill-climb with plateau moves. ``times1`` are 1-based layers on ``inst``
    (edge-aligned with ``graph.edges``). Returns ``(best_times1, best_score, log)``.

    A move is accepted if the witness weight rises, or with probability
    ``plateau`` if it stays equal (the witness changes, which is progress on a
    plateau). Only edges implicated by the current witness are moved.
    """
    rng = np.random.default_rng(seed)
    T = int(times1.max())
    ends = [(("q", q), ck) for ck, q, _ty in graph.edges]

    def index(times):
        at = {}
        for e, (u, v) in enumerate(ends):
            at[(u, int(times[e]))] = e
            at[(v, int(times[e]))] = e
        return at

    cur = times1.copy()
    cur_s, per = score(code, cur - 1, graph, lim=lim, p=p)
    best, best_s = cur.copy(), cur_s
    log = [(0, cur_s)]
    rejected = evaluated = 0
    t0 = time.time()
    if verbose:
        print(f"    start: witness weight {cur_s} "
              f"(Z={per['Z'][0]}, X={per['X'][0]})", flush=True)
    for it in range(1, iters + 1):
        faults = [f for b in ("Z", "X") for f in per[b][1]
                  if per[b][0] is not None and per[b][0] == cur_s] or \
                 [f for b in ("Z", "X") for f in per[b][1]]
        T_edges = sorted(touched_edges(code, graph, cur - 1, faults))
        if not T_edges:
            break
        e = int(rng.choice(T_edges))
        t1 = int(cur[e])
        t2 = int(rng.choice([t for t in range(1, T + 1) if t != t1]))
        chain = _kempe(e, t2, cur, ends, index(cur))
        cand = cur.copy()
        for f in chain:
            cand[f] = t2 if cur[f] == t1 else t1
        if not _valid(inst, cand, T):
            rejected += 1
            continue
        s, per2 = score(code, cand - 1, graph, lim=lim, p=p)
        evaluated += 1
        if s is None:
            continue
        accept = s > cur_s or (s == cur_s and rng.random() < plateau)
        if accept:
            cur, cur_s, per = cand, s, per2
            if s > best_s:
                best, best_s = cand.copy(), s
                log.append((it, s))
                if verbose:
                    print(f"    iter {it}: NEW BEST {s} "
                          f"(Z={per2['Z'][0]}, X={per2['X'][0]}) "
                          f"({time.time() - t0:.0f}s)", flush=True)
        if verbose and it % 50 == 0:
            print(f"    iter {it}: current {cur_s}, best {best_s}; "
                  f"{evaluated} evaluated, {rejected} rejected by verify "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if verbose:
        print(f"    done: {evaluated} evaluated, {rejected} rejected by verify",
              flush=True)
    return best, best_s, log
