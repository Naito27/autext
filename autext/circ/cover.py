"""Exhaustive minimum-weight search on a DEM by cover branching.

Why not CP-SAT.  Branching on the mechanism variables scales steeply in the
weight cap: each extra unit of weight multiplies the cost by a large factor, so
a weight-10 refutation on a DEM with hundreds of mechanisms is out of reach.
The structure the variable branching throws away is that ``D f = 0`` is a COVER
condition --- if a detector's parity is open, one of the mechanisms touching it
has to be in the set.  Branching on that gives depth ``<= w`` and branching
factor ``<= max detector degree``, with the running syndrome as the propagator.

The search, seeded at one mechanism:

  * ``syndrome == 0`` --- a solution iff some logical flipped;
  * otherwise take the LOWEST open detector and branch over the mechanisms
    touching it, the ``i``-th branch choosing ``m_i`` and excluding
    ``m_0..m_{i-1}``.  Canonical: every set is reached once.

Two levels are closed by table instead of by branching: with one mechanism
left the syndrome must BE some column, and with two left it must be some
pair's XOR.  The pair table is ``C(nE, 2)`` entries, built once.  That removes
the two most populous levels of the tree.  It is exact because the ladder makes
the completion length exact, and ``tables=False`` turns it off so that the two
routes can be cross-checked.

Prunes: the weight budget; ``popcount(syndrome) <= budget * maxdeg`` because
each further mechanism closes at most ``maxdeg`` open detectors; a greedy
INDEPENDENT SET among the open detectors, since two open detectors that no
single mechanism touches together need two different mechanisms (much sharper
than the popcount bound wherever the syndrome is spread out); and a node whose
chosen detector has no usable mechanism left.  ``mrv`` branches on the
open detector with the fewest usable mechanisms rather than the lowest one:
any deterministic choice keeps the enumeration canonical, and the smallest
branching factor shrinks the tree the most.

``exclude`` is the second symmetry cut.  Fix any order on the orbits.  For a
solution ``S`` let ``o`` be the EARLIEST orbit it meets; translating some
element of ``S`` in that orbit onto the orbit's representative gives a
solution that contains the representative and still meets no earlier orbit.
So running seed ``o`` with every mechanism of an earlier orbit in ``exclude``
is complete, and the later seeds search a far smaller space.

**Completeness at the minimum weight.**  Let ``S`` be a minimum-weight
solution.  No proper subset ``A`` of ``S`` reached along the branch can be
closed: if ``A`` flips a logical it is a lighter solution, and if it does not
then ``S \\ A`` is one.  So the branch following ``S`` never stops early and
reaches ``S``.  Running the ladder upwards from ``w = 1`` means that at cap
``w`` every lighter weight is already refuted, so a completion needs EXACTLY
the remaining budget --- which is what makes the two table levels exact.

**Seeding.**  ``S`` contains some mechanism, and a symmetry of the circuit
maps solutions to solutions, so some translate of ``S`` contains an orbit
representative: the representatives are a complete set of seeds.  Pass
``reps=None`` to seed at every mechanism instead (always sound, ``|orbit|``
times slower).
"""
from __future__ import annotations

import itertools
import time

import numpy as np

__all__ = ["CoverSearch", "cover_ladder", "find_witness"]


class _Timeout(Exception):
    pass


class CoverSearch:
    """Cover-branching search over a DEM ``(D, O)``."""

    def __init__(self, D, O, *, pairs=True, tables=True, mrv=True):
        D = np.asarray(D, dtype=np.uint8)
        O = np.asarray(O, dtype=np.uint8)
        self.nD, self.nE = D.shape
        self.cols = []
        self.logs = []
        for e in range(self.nE):
            v = 0
            for d in np.nonzero(D[:, e])[0]:
                v |= 1 << int(d)
            self.cols.append(v)
            g = 0
            for i in np.nonzero(O[:, e])[0]:
                g |= 1 << int(i)
            self.logs.append(g)
        self.inc = [tuple(int(e) for e in np.nonzero(D[d])[0]) for d in range(self.nD)]
        # detectors reachable from d through ONE mechanism (d included).  Two
        # open detectors outside each other's mask need two different
        # mechanisms, so a greedy independent set among the open detectors is a
        # lower bound on how many are still required -- far sharper than
        # popcount/maxdeg wherever the syndrome is spread out.
        self.nbr = []
        for d in range(self.nD):
            msk = 1 << d
            for e in self.inc[d]:
                msk |= self.cols[e]
            self.nbr.append(msk)
        self.nnbr = [~v for v in self.nbr]   # complement of nbr, precomputed:
        # the independent-set loop needs it at every step of every node.
        self.dsup = [tuple(int(d) for d in np.nonzero(D[:, e])[0])
                     for e in range(self.nE)]
        self.cnt = [0] * self.nD             # usable mechanisms per detector,
        # maintained incrementally rather than recounted at every node
        self.maxdeg = int(D.sum(axis=0).max()) or 1
        self.tables = tables
        self.mrv = mrv               # branch on the open detector with FEWEST candidates
        self.is_from = 3             # independent-set bound only where we
                                     # BRANCH (rem >= 3): at the table-closed
                                     # levels it costs more than it saves
        self.exclude = ()            # mechanisms barred outright (the orbit-prefix cut)
        self.find_leq = False
        # UPPER-BOUND MODE.  ``seed`` normally assumes every lighter weight has
        # already been refuted, which is what licenses the ``syn == 0`` prune
        # ("a lighter solution -- already refuted, so not a new one").  With
        # ``find_leq`` that prune instead ACCEPTS the lighter solution, so a hit
        # at cap ``w`` means "some solution of weight <= w exists" and no ladder
        # below ``w`` is needed.  That is sound for an upper bound and UNSOUND
        # for a refutation: a miss in this mode proves nothing, because the
        # canonical ``forb`` exclusion is only exhaustive per exact weight.
        # Callers must never read a False from this mode as a refutation.
        self.single = {}
        for e, v in enumerate(self.cols):
            self.single.setdefault(v, []).append(e)
        self.pair = None
        if pairs and tables:
            p: dict = {}
            cols = self.cols
            for i, j in itertools.combinations(range(self.nE), 2):
                p.setdefault(cols[i] ^ cols[j], []).append((i, j))
            self.pair = p
        self.forb = bytearray(self.nE)
        self.nodes = 0
        self.witness = None          # the mechanism set of the last solution found
        self._path = []
        self._deadline = float("inf")

    # ------------------------------------------------------------------ search
    def _dfs(self, syn, lg, rem):
        """Is there a completion of EXACTLY ``rem`` more mechanisms?"""
        self.nodes += 1
        if not (self.nodes & 0xFFFF) and time.time() > self._deadline:
            raise _Timeout
        if rem == 0:
            if syn == 0 and lg != 0:
                self.witness = list(self._path)
                return True
            return False
        if syn == 0:
            if self.find_leq and lg != 0:
                self.witness = list(self._path)
                return True
            return False                      # lighter solution: refuted at a lower cap
        if syn.bit_count() > rem * self.maxdeg:
            return False
        if rem >= self.is_from:
            # The independent-set bound costs roughly twice an ordinary node,
            # so paying it at the most populous (deepest) levels loses more
            # than it saves: fewer nodes, but longer overall.  Apply it only
            # near the top, where one prune kills a large subtree.
            nnbr = self.nnbr
            left, need = syn, 0
            while left:
                d0 = (left & -left).bit_length() - 1
                need += 1
                if need > rem:
                    return False
                left &= nnbr[d0]
        forb, logs = self.forb, self.logs
        if rem == 1 and self.tables:
            for m in self.single.get(syn, ()):
                if not forb[m] and (lg ^ logs[m]) != 0:
                    self.witness = self._path + [m]
                    return True
            return False
        if rem == 2 and self.pair is not None:
            for m1, m2 in self.pair.get(syn, ()):
                if not forb[m1] and not forb[m2] and (lg ^ logs[m1] ^ logs[m2]) != 0:
                    self.witness = self._path + [m1, m2]
                    return True
            return False
        d = self._pick(syn)
        if d is None:
            return False              # an open detector no remaining mechanism can close
        cols = self.cols
        cnt, dsup = self.cnt, self.dsup
        changed = []
        hit = False
        for m in self.inc[d]:
            if forb[m]:
                continue
            forb[m] = 1
            for dd in dsup[m]:
                cnt[dd] -= 1
            changed.append(m)
            self._path.append(m)
            if self._dfs(syn ^ cols[m], lg ^ logs[m], rem - 1):
                hit = True
                self._path.pop()
                break
            self._path.pop()
        for m in changed:
            forb[m] = 0
            for dd in dsup[m]:
                cnt[dd] += 1
        return hit

    def _reset_forb(self, r):
        """Clear ``forb``, bar the excluded set and ``r``, and rebuild ``cnt``."""
        self.forb[:] = bytearray(self.nE)
        for i in self.exclude:
            self.forb[i] = 1
        self.forb[r] = 1
        forb = self.forb
        cnt = self.cnt
        for d in range(self.nD):
            c = 0
            for mm in self.inc[d]:
                if not forb[mm]:
                    c += 1
            cnt[d] = c

    def root_branches(self, r, w):
        """The first-level choices for ``seed(r, w)``: how many, deterministically."""
        self._reset_forb(r)
        syn = self.cols[r]
        if syn == 0 or w - 1 <= 2:
            return 1                       # nothing to split: one unit of work
        d = self._pick(syn)
        if d is None:
            return 0
        return sum(1 for m in self.inc[d] if not self.forb[m])

    def _pick(self, syn):
        """The detector to branch on: fewest usable mechanisms, else the lowest."""
        if not self.mrv:
            return (syn & -syn).bit_length() - 1
        cnt = self.cnt
        best, bd = None, 1 << 30
        x = syn
        while x:
            low = x & -x
            dd = low.bit_length() - 1
            c = cnt[dd]
            if c < bd:
                bd, best = c, dd
                if c <= 1:
                    break
            x ^= low
        return None if bd == 0 else best

    def seed(self, r, w, time_limit=float("inf"), only_branch=None):
        """Is there a weight-``w`` solution containing mechanism ``r``?

        Assumes every weight ``< w`` has already been refuted.  Returns
        ``True`` / ``False``, or raises :class:`TimeoutError`.

        ``only_branch`` explores just that one first-level choice, so a single
        heavy seed can be split across workers; the union over
        ``range(root_branches(r, w))`` is the whole seed.
        """
        self._reset_forb(r)
        self.nodes = 0
        self.witness = None
        self._path = [r]
        self._deadline = time.time() + time_limit
        try:
            if only_branch is None:
                return self._dfs(self.cols[r], self.logs[r], w - 1)
            syn, lg, rem = self.cols[r], self.logs[r], w - 1
            if syn == 0 or rem <= 2:
                return self._dfs(syn, lg, rem) if only_branch == 0 else False
            d = self._pick(syn)
            if d is None:
                return False
            k = 0
            for mm in self.inc[d]:
                if self.forb[mm]:
                    continue
                if k == only_branch:
                    self.forb[mm] = 1
                    for dd in self.dsup[mm]:
                        self.cnt[dd] -= 1
                    self._path.append(mm)
                    return self._dfs(syn ^ self.cols[mm], lg ^ self.logs[mm], rem - 1)
                self.forb[mm] = 1          # earlier branches: excluded in this subtree
                for dd in self.dsup[mm]:
                    self.cnt[dd] -= 1
                k += 1
            return False
        except _Timeout:
            raise TimeoutError(f"cover search timed out at seed {r}, cap {w}")


def _find_dfs(cs, syn, lg, rem, chosen, budget, rng, greedy):
    """Randomised greedy DFS for ANY fault set of weight ``<= rem`` more.

    Not the canonical enumeration: this is a FINDER.  It may reach the same set
    by several routes and it gives up when the node budget runs out, so a
    failure proves nothing -- but anything it returns is a genuine solution,
    and it exploits the cover structure that a decoder like BP-OSD cannot see.
    """
    budget[0] -= 1
    if budget[0] <= 0:
        return None
    if syn == 0:
        return list(chosen) if lg != 0 else None
    if rem == 0 or syn.bit_count() > rem * cs.maxdeg:
        return None
    nbr = cs.nbr
    left, need = syn, 0
    while left:
        d0 = (left & -left).bit_length() - 1
        need += 1
        if need > rem:
            return None
        left &= ~nbr[d0]
    # the open detector with the fewest usable mechanisms
    best, bd = None, 1 << 30
    left = syn
    while left:
        low = left & -left
        dd = low.bit_length() - 1
        cnt = sum(1 for mm in cs.inc[dd] if mm not in chosen)
        if cnt < bd:
            bd, best = cnt, dd
            if cnt <= 1:
                break
        left ^= low
    if bd == 0:
        return None
    cand = [mm for mm in cs.inc[best] if mm not in chosen]
    cols, logs = cs.cols, cs.logs
    if greedy:                      # prefer the mechanism that closes the most
        cand.sort(key=lambda mm: (syn ^ cols[mm]).bit_count() + rng.random())
    else:
        rng.shuffle(cand)
    for mm in cand:
        chosen.add(mm)
        out = _find_dfs(cs, syn ^ cols[mm], lg ^ logs[mm], rem - 1, chosen,
                        budget, rng, greedy)
        chosen.discard(mm)
        if out is not None:
            return out
    return None


def find_witness(cs, w, *, tries=40, node_budget=20000, seed=0, seeds=None):
    """Cheap search for SOME fault set of weight ``<= w``; ``None`` if not found.

    ``None`` is not a refutation.  Returns the mechanism list of a real
    solution, which the caller should still verify against the DEM.
    """
    import random
    rng = random.Random(seed)
    starts = list(seeds) if seeds is not None else list(range(cs.nE))
    for t in range(tries):
        r = starts[rng.randrange(len(starts))]
        out = _find_dfs(cs, cs.cols[r], cs.logs[r], w - 1, {r},
                        [node_budget], rng, greedy=(t % 2 == 0))
        if out is not None:
            return out
    return None


def orbit_of(E, perms):
    """``orbit[m]`` for every mechanism, orbits numbered by their smallest member."""
    orbit = [-1] * E
    nxt = 0
    for e in range(E):
        if orbit[e] >= 0:
            continue
        stack = [e]
        orbit[e] = nxt
        while stack:
            x = stack.pop()
            for p in perms:
                y = int(p[x])
                if orbit[y] < 0:
                    orbit[y] = nxt
                    stack.append(y)
        nxt += 1
    return orbit, nxt


def orbit_seeds(E, perms):
    """``[(representative, exclusion tuple), ...]`` implementing the orbit-prefix cut."""
    orbit, north = orbit_of(E, perms)
    members = {o: [e for e in range(E) if orbit[e] == o] for o in range(north)}
    order = sorted(range(north), key=lambda o: members[o][0])
    out, acc = [], []
    for o in order:
        out.append((members[o][0], tuple(acc)))
        acc += members[o]
    return out


def cover_ladder(D, O, reps, wmax, *, time_limit=float("inf"), log=print, wmin=1):
    """Refute weights ``wmin..wmax`` in turn; stop at the first that is met.

    Returns ``(w, seed)`` for the first weight with a solution, or
    ``(wmax + 1, None)`` if every weight up to ``wmax`` is refuted --- a proof
    that ``d_circ >= wmax + 1`` for this DEM.
    """
    cs = CoverSearch(D, O)
    t_all = time.time()
    for w in range(max(1, wmin), wmax + 1):
        t0 = time.time()
        total = 0
        for r in reps:
            hit = cs.seed(r, w, time_limit=max(0.0, t_all + time_limit - time.time()))
            total += cs.nodes
            if hit:
                log(f"  cap {w:2d}: SOLUTION containing mechanism {r} "
                    f"({total:,} nodes, {time.time() - t0:.1f}s)")
                return w, r
        log(f"  cap {w:2d}: refuted ({total:,} nodes, {time.time() - t0:.1f}s)")
    return wmax + 1, None
