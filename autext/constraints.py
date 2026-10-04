"""The two scheduling constraints on a syndrome-extraction circuit.

**(C1) Resource / proper colouring.** Each ancilla and data qubit does at most
one CX per layer — the schedule is a proper edge colouring of the Tanner graph.
By König's theorem a bipartite graph of max degree ``Delta`` is
``Delta``-edge-colourable, so (C1) alone never forces depth above ``Delta``.

**(C2) Contamination-freeness.** For every overlapping ``(X-check a, Z-check b)``
the shared-qubit CX gates must be ordered so the X-CX precedes the Z-CX an
*even* number of times:

    c(a, b) = sum_{q in supp(a) cap supp(b)} 1[tau(a,q) < tau(b,q)] == 0 (mod 2).

CSS commutation forces the overlap even, so an even leak deposits a stabilizer
(harmless); odd is a genuine fault propagating between ancillas.
"""
from __future__ import annotations

from .core import Edge, TannerGraph

__all__ = ["bipartite_edge_color", "verify_proper", "contamination_ok"]


# --------------------------------------------------------------------------
# (C1) proper edge colouring (König, alternating-path)
# --------------------------------------------------------------------------
def _endpoints(edge: Edge) -> tuple:
    ck, q, _ty = edge
    return ("C", ck), ("Q", q)


def bipartite_edge_color(edges: list[Edge], delta: int) -> list[int]:
    """Proper edge colouring with colours ``0..delta-1`` (König, alternating-path)."""
    inc: dict[tuple, dict[int, int]] = {}
    color = [-1] * len(edges)
    for e in range(len(edges)):
        u, v = _endpoints(edges[e])
        inc.setdefault(u, {})
        inc.setdefault(v, {})
        free_u = [c for c in range(delta) if c not in inc[u]]
        free_v = [c for c in range(delta) if c not in inc[v]]
        common = set(free_u) & set(free_v)
        if common:
            col = min(common)
        else:
            alpha = min(free_u)  # free at u, used at v
            beta = min(free_v)   # free at v, used at u
            _flip(edges, inc, color, v, alpha, beta)
            col = alpha
        color[e] = col
        inc[u][col] = e
        inc[v][col] = e
    return color


def _flip(edges, inc, color, start, alpha, beta) -> None:
    """Flip the alpha/beta colours along the maximal alternating path from ``start``."""
    cur, cc, chain = start, alpha, []
    while cur in inc and cc in inc[cur]:
        e = inc[cur][cc]
        chain.append((e, cc))
        a, b = _endpoints(edges[e])
        cur = b if a == cur else a
        cc = beta if cc == alpha else alpha
    for e, oldc in chain:
        a, b = _endpoints(edges[e])
        if inc[a].get(oldc) == e:
            del inc[a][oldc]
        if inc[b].get(oldc) == e:
            del inc[b][oldc]
    for e, oldc in chain:
        newc = beta if oldc == alpha else alpha
        a, b = _endpoints(edges[e])
        color[e] = newc
        inc[a][newc] = e
        inc[b][newc] = e


def verify_proper(edges: list[Edge], times, delta: int) -> bool:
    """True iff ``times`` is a proper colouring with values in ``[0, delta)``."""
    seen: dict[tuple, int] = {}
    for e, edge in enumerate(edges):
        for vert in _endpoints(edge):
            key = (vert, times[e])
            if key in seen:
                return False
            seen[key] = e
    return all(0 <= t < delta for t in times)


# --------------------------------------------------------------------------
# (C2) contamination-freeness (net-leak parity formula)
# --------------------------------------------------------------------------
def contamination_ok(times, graph: TannerGraph) -> bool:
    """True iff every overlapping X/Z pair has even net-leak parity."""
    eidx = graph.eidx
    for (a, b), shared in graph.overlaps.items():
        cnt = 0
        for q in shared:
            ex = eidx[(("X", a), q)]
            ez = eidx[(("Z", b), q)]
            if times[ex] < times[ez]:
                cnt += 1
        if cnt % 2:
            return False
    return True
