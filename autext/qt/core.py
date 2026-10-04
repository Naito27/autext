"""Shared machinery for the bit-oriented scheduling problem.

`Inst` is the instance type every solver and every check script speaks: per-edge
square, check and group (1 = early, 2 = middle, 3 = late in the sandwich of the
canonical orientation).  `verify` is the referee for a candidate schedule.
`straddles` and `layer_hall` are two necessary conditions for depth T; a
halving failing `straddles` admits no depth-Delta schedule at all.
"""
import numpy as np


class Inst:
    """Base-graph instance: per-edge square, check and group (1,2,3)."""

    def __init__(self, sq, ck, grp, nsq, nck):
        # int32 halves the largest arrays; every product below is widened
        # explicitly, since NEP 50 keeps int32 * python-int in int32
        self.sq = np.asarray(sq, np.int32)
        self.ck = np.asarray(ck, np.int32)
        self.grp = np.asarray(grp, np.int8)
        self.nsq, self.nck = int(nsq), int(nck)
        self.m = len(self.sq)
        self.degS = np.bincount(self.sq, minlength=nsq)
        self.degC = np.bincount(self.ck, minlength=nck)
        # bincount on the flattened (square, group) key; np.add.at is an
        # unbuffered scatter and is far slower on millions of edges
        self.ng = np.bincount(self.sq.astype(np.int64) * 4 + self.grp,
                              minlength=nsq * 4).reshape(nsq, 4)
        self._sq_list = self._ck_list = None

    def Delta(self):
        return int(max(self.degS.max(), self.degC.max()))

    @property
    def sq_list(self):
        """`sq` as a Python list, built once and cached.

        The solver's inner loop is pure Python, where indexing a list of ints is
        several times cheaper than indexing an int32 array (which boxes a numpy
        scalar).  `run` is called once per (seed, depth) on the same instance,
        so this is built once instead of once per call."""
        if self._sq_list is None:
            self._sq_list = self.sq.tolist()
        return self._sq_list

    @property
    def ck_list(self):
        """`ck` as a Python list, built once and cached.  See `sq_list`."""
        if self._ck_list is None:
            self._ck_list = self.ck.tolist()
        return self._ck_list


def verify(inst, sched, T):
    assert (sched >= 1).all() and (sched <= T).all(), 'range'
    for vid in (inst.sq, inst.ck):
        key = vid.astype(np.int64) * (T + 1) + sched
        assert len(np.unique(key)) == len(key), 'collision'
    order = np.lexsort((sched, inst.sq))
    s, g = inst.sq[order], inst.grp[order]
    t = sched[order]
    same = s[1:] == s[:-1]
    assert not np.any(same & (g[1:] < g[:-1])), 'precedence violated'
    assert not np.any(same & (g[1:] == g[:-1]) & (t[1:] == t[:-1])), 'dup'
    return True


def straddles(inst, T=None):
    """Straddling test.  A check of degree T must fire in layer 1 and in
    layer T, and at a square v an edge can carry layer 1 only if its group is
    the FIRST NON-EMPTY group at v (not necessarily the early family: the early
    family may have degree zero there), and layer T only if its group is the
    LAST NON-EMPTY group."""
    T = inst.Delta() if T is None else T
    first, last = first_last_group(inst)
    tight = np.flatnonzero(inst.degC == T)
    if len(tight) == 0:
        return True
    isfirst = inst.grp == first[inst.sq]
    islast = inst.grp == last[inst.sq]
    # bincount, not np.logical_or.at: the .at ufuncs are unbuffered scatters
    # and are far slower on millions of edges
    okf = np.bincount(inst.ck, weights=isfirst, minlength=inst.nck) > 0
    okl = np.bincount(inst.ck, weights=islast, minlength=inst.nck) > 0
    return bool(okf[tight].all() and okl[tight].all())


def windows(inst, T):
    """The window lemma.  In any depth-T schedule an edge e of group g at square
    v can only take a layer in

        W_e = [ 1 + sum_{i<g} n_i(v) ,  T - sum_{i>g} n_i(v) ]      (1-indexed)

    where n_i(v) is the number of edges of group i at v: the edges of earlier
    groups at v occupy that many distinct earlier layers (one edge per layer per
    square), and symmetrically for the later groups.  Returns the per-edge
    arrays (lo, hi), inclusive.

    This is a sound restriction, never a constraint of the problem itself: the
    problem is defined by the two collision conditions and the group order, and
    W_e is implied by them.  `layer_hall` uses it to build the per-layer
    eligible subgraph; `exact` uses it to shrink CP-SAT variable domains."""
    ng = inst.ng
    cum = np.cumsum(ng, axis=1)
    pre = cum - ng
    post = ng.sum(1)[:, None] - cum
    return 1 + pre[inst.sq, inst.grp], T - post[inst.sq, inst.grp]


def layer_hall(inst, T):
    """Layerwise Hall: for every layer, the window graph must contain a matching
    saturating every square and every check of degree T.  Necessary for ANY
    depth-T schedule, so a failure is a proof of infeasibility."""
    if inst.Delta() > T:
        return False, ('degree', None, None)
    lo, hi = windows(inst, T)
    tS = np.flatnonzero(inst.degS == T)
    tC = np.flatnonzero(inst.degC == T)
    for t in range(1, T + 1):
        live = np.flatnonzero((lo <= t) & (t <= hi))
        adjS, adjC = {}, {}
        for e in live:
            adjS.setdefault(int(inst.sq[e]), []).append(int(inst.ck[e]))
            adjC.setdefault(int(inst.ck[e]), []).append(int(inst.sq[e]))
        for adj, need, name in ((adjS, tS, 'square'), (adjC, tC, 'check')):
            match = {}

            def aug(u, seen):
                for w in adj.get(u, ()):
                    if w in seen:
                        continue
                    seen.add(w)
                    if w not in match or aug(match[w], seen):
                        match[w] = u
                        return True
                return False

            for u in need:
                if not aug(int(u), set()):
                    return False, (name, t, int(u))
    return True, None


def first_last_group(inst):
    """First and last NON-EMPTY group at each square."""
    ng = inst.ng[:, 1:]                       # groups 1,2,3
    has = ng > 0
    first = np.where(has.any(1), has.argmax(1) + 1, 0)
    last = np.where(has.any(1), 3 - has[:, ::-1].argmax(1), 0)
    return first, last
