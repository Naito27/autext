"""The simplified schedule: Koenig's algorithm on the eligible subgraph.

    for t = 1 .. T:
        E_t  = unscheduled edges whose group is CURRENT at their square
        crit = vertices whose remaining degree equals the budget T-t+1
        M    = a matching in E_t saturating crit, extended greedily
        schedule M at layer t and delete it

This is the paper's Box 1.  No windows, no deadlines, no propagation, no forced
edges, no release move.  The precedence needs no enforcement: a square never
offers a later-group edge while an earlier-group edge survives, so its group-1
layers all precede its group-2 layers, and so on.  Strip the ordering and
"eligible" becomes "unscheduled" and this is exactly Koenig -- delete a matching
saturating every maximum-degree vertex, the maximum degree drops by one --
which is why the target is Delta.

The matching is built in three stages per layer: augment for every critical
vertex (the part Koenig needs), a greedy linear pass over the eligible edges in
urgency order, then augment the still-unmatched squares to make the matching
maximum.  That last stage is not needed for a valid schedule, and `enlarge` can
turn it off.  Whether it pays depends on size: on the small base pairs it is
what reaches Delta, but it is O(V x E) per layer and dominates the runtime on
large instances, where the cheap pass reaches Delta anyway.  `solve` therefore
tries `enlarge=False` first and falls back.

The augmenting search dominates, so it is written without any allocation: the
per-layer adjacency is CSR (a flat edge array plus offsets, grouped by a stable
argsort so the within-vertex order is still urgency order), and the DFS stack
and alternating trail are preallocated Python lists indexed by a top pointer.
Vertex ids come from `Inst.sq_list`/`ck_list`, cached on the instance rather
than rebuilt per call.  None of this changes a decision the solver makes.
"""
import numpy as np
from .core import verify                       # the one referee


def run(inst, T, seed=0, trace=False, enlarge=True):
    rng = np.random.default_rng(seed)
    sq_np, ck_np, grp = inst.sq, inst.ck, inst.grp
    SQ = inst.sq_list; CK = inst.ck_list
    nsq, nck = inst.nsq, inst.nck
    alive = np.ones(inst.m, bool)
    ng = inst.ng.copy()
    degS = inst.degS.copy(); degC = inst.degC.copy()
    if max(degS.max(), degC.max()) > T:
        return None
    sched = np.full(inst.m, -1, np.int64)
    mS = [-1] * nsq; mC = [-1] * nck
    seenC = [0] * nck; seenS = [0] * nsq
    stamp = 0

    # preallocated DFS scratch: an alternating path visits each square at most
    # once, so nsq + 1 frames is always enough
    cap = nsq + 1
    st_v = [0] * cap; st_p = [0] * cap
    tr_v = [0] * cap; tr_e = [0] * cap; tr_c = [0] * cap

    # per-layer CSR adjacency, rebound each layer
    adjS = adjC = None
    offS = offC = None

    def aug_sq(v):
        """Augmenting DFS from square `v` over the eligible subgraph.

        The visited marks are NOT reset per call, only when the matching
        changes.  A failed search proves that nothing reachable from the checks
        it visited is free; the matching has not moved, so that stays true for
        every later search in the same layer, and skipping those checks cannot
        lose an augmenting path.  This is what makes the enlargement phase
        linear in the layer rather than quadratic."""
        nonlocal stamp
        top = 0; st_v[0] = v; st_p[0] = offS[v]
        tlen = 0
        while top >= 0:
            u = st_v[top]; p = st_p[top]; end = offS[u + 1]
            pushed = False
            while p < end:
                e = adjS[p]; p += 1
                c = CK[e]
                if seenC[c] == stamp:
                    continue
                seenC[c] = stamp
                tr_v[tlen] = u; tr_e[tlen] = e; tr_c[tlen] = c; tlen += 1
                st_p[top] = p
                if mC[c] < 0:
                    for i in range(tlen):
                        mS[tr_v[i]] = tr_e[i]; mC[tr_c[i]] = tr_e[i]
                    stamp += 1          # matching moved: marks are now stale
                    return True
                w = SQ[mC[c]]
                top += 1; st_v[top] = w; st_p[top] = offS[w]
                pushed = True
                break
            if not pushed:
                st_p[top] = p
                top -= 1
                if tlen:
                    tlen -= 1
        return False

    def aug_ck(c0):
        """The same search from a check, over the transposed adjacency."""
        nonlocal stamp
        top = 0; st_v[0] = c0; st_p[0] = offC[c0]
        tlen = 0
        while top >= 0:
            u = st_v[top]; p = st_p[top]; end = offC[u + 1]
            pushed = False
            while p < end:
                e = adjC[p]; p += 1
                v = SQ[e]
                if seenS[v] == stamp:
                    continue
                seenS[v] = stamp
                tr_v[tlen] = u; tr_e[tlen] = e; tr_c[tlen] = v; tlen += 1
                st_p[top] = p
                if mS[v] < 0:
                    for i in range(tlen):
                        mS[tr_c[i]] = tr_e[i]; mC[tr_v[i]] = tr_e[i]
                    stamp += 1          # matching moved: marks are now stale
                    return True
                w = CK[mS[v]]
                top += 1; st_v[top] = w; st_p[top] = offC[w]
                pushed = True
                break
            if not pushed:
                st_p[top] = p
                top -= 1
                if tlen:
                    tlen -= 1
        return False

    for t in range(1, T + 1):
        if not alive.any():
            return sched
        cur = np.where(ng[:, 1] > 0, 1, np.where(ng[:, 2] > 0, 2, 3))
        elig = np.flatnonzero(alive & (grp == cur[sq_np]))
        if len(elig) == 0:
            return None
        urg = degS[sq_np[elig]] + degC[ck_np[elig]] + rng.random(len(elig))
        elig = elig[np.argsort(-urg, kind='stable')]

        # CSR by vertex.  A STABLE argsort on the vertex id groups the edges
        # while leaving each group in urgency order, which is exactly the order
        # the list-of-lists version appended in.
        se = sq_np[elig]; ce = ck_np[elig]
        oS = np.argsort(se, kind='stable'); oC = np.argsort(ce, kind='stable')
        adjS = elig[oS].tolist(); adjC = elig[oC].tolist()
        cntS = np.bincount(se, minlength=nsq); cntC = np.bincount(ce, minlength=nck)
        offS = np.concatenate(([0], np.cumsum(cntS))).tolist()
        offC = np.concatenate(([0], np.cumsum(cntC))).tolist()
        # `touchedS` must stay in FIRST-ENCOUNTER order, not ascending id: the
        # enlargement below sorts it by -degS with a STABLE sort, so this order
        # breaks the ties and decides which square is augmented first.
        uS, firstS = np.unique(se, return_index=True)
        touchedS = uS[np.argsort(firstS)].tolist()
        touchedC = np.flatnonzero(cntC).tolist()     # reset order only
        elig_l = elig.tolist()

        stamp += 1                    # fresh marks for this layer
        beta = T - t + 1
        fail = False
        for v in np.flatnonzero(degS == beta).tolist():          # critical squares
            if mS[v] < 0 and not aug_sq(v):
                fail = True; break
        if not fail:
            for c in np.flatnonzero(degC == beta).tolist():      # critical checks
                if mC[c] < 0 and not aug_ck(c):
                    fail = True; break
        if not fail:
            for e in elig_l:                                     # enlarge: linear pass
                v = SQ[e]
                if mS[v] < 0 and mC[CK[e]] < 0:
                    mS[v] = e; mC[CK[e]] = e
            if enlarge:
                for v in sorted((v for v in touchedS if mS[v] < 0),
                                key=lambda v: -degS[v]):          # then augment
                    if mS[v] < 0:
                        aug_sq(v)
        chosen = [mS[v] for v in touchedS if mS[v] >= 0]
        for v in touchedS:
            mS[v] = -1
        for c in touchedC:
            mC[c] = -1
        if fail:
            return None
        ch = np.array(chosen, np.int64)
        if trace:
            print(f'   t={t:4d} |M|={len(ch):5d} alive={int(alive.sum()):6d}')
        alive[ch] = False; sched[ch] = t
        np.subtract.at(ng, (sq_np[ch], grp[ch]), 1)
        np.subtract.at(degS, sq_np[ch], 1)
        np.subtract.at(degC, ck_np[ch], 1)
    return sched if not alive.any() else None


def solve(inst, span=3, seeds=8):
    """Smallest depth reached, with the schedule.

    Each depth is tried WITHOUT the enlargement first and with it only if that
    misses.  The cheap pass is what succeeds at scale (the enlargement is
    O(V x E) per layer and dominates the runtime on large instances); the
    thorough pass is what the small hard cases need.  Trying cheap-then-
    thorough can only reach the same depth or better than the thorough pass
    alone, and never costs more than a constant factor extra when it fails."""
    D = inst.Delta()
    for k in range(span):
        for enl in (False, True):
            for s in range(seeds):
                sch = run(inst, D + k, seed=s, enlarge=enl)
                if sch is not None:
                    verify(inst, sch, D + k)
                    return D + k, sch
    return None, None
