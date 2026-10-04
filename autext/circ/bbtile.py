"""Tiled sandwich circuits of a BB code, pruned by explicit light logicals of
the extended code (a counterexample-guided loop).

The idea (Strikis, Browne and Beverland, arXiv:2603.05481, Prop. 3).
``d_circ >= d_ext(E)`` for ANY circuit, where ``E`` is its residual set and
``E`` depends only on each check's own CNOT order.  A logical of the extended
code of weight ``<= 10`` that uses residual columns ``E_1..E_k`` is therefore
a reusable NO-GOOD: every circuit whose orders produce all of ``E_1..E_k`` has
``d_ext <= 10`` (monotonicity: more columns can only lower ``d_ext``), and so
does every translate of that pattern, because translations are automorphisms
of the code.  The loop alternates a small CP-SAT scheduling model (the
"reduced model") with a motif search on its candidate, until the model is
infeasible (certificate: nothing in the family can be certified ``>= 11``
this way) or a candidate survives the motif search (a SURVIVOR, to be
attacked with the expensive proof solve).

The family: **sandwich** circuits of depth ``D`` (bands early / middle /
late, ``a`` and ``a*`` edges in the middle band, ``b`` and ``b*`` edges
outside it), **tiled** by a colouring: checks of one class share their CNOT
order.  The colouring is ANY map from checks to classes that is proper on
each check type's overlap graph (X and Z checks may be coloured
independently): a linear form mod 3 (Strikis et al.'s three-colour tiling), a
product of linear forms (finer periodic tilings), or a random proper
colouring (non-periodic).  Uniform tiling is the one-class colouring.
Properness is imposed as the parity constraint on every overlapping pair, so
the family is exactly "proper sandwich-band circuits with per-class orders".

Residual columns follow the ``dext`` convention: a check order
``q1..q6`` contributes ``{q1,q2}``, ``{q4,q5,q6}`` (equivalent to
``{q1,q2,q3}`` modulo the check) and ``{q5,q6}``; the other suffixes are
single data qubits modulo the check.  ``d_ext`` is unchanged from the
five-suffix form used elsewhere in the package.

Conventions match :mod:`autext.lp_sandwich`: ``HX = [A | B]``,
``HZ = [B^T | A^T]``; an X-edge's TYPE is its monomial index (``0..2`` for
``a_k``, ``3..5`` for ``b_k``) and a Z-edge's is ``0..2`` for ``a*_k`` and
``3..5`` for ``b*_k``.  Check ``cid`` is the group element ``x^i y^j`` with
``cid = i*m + j``, and translation by ``(di, dj)`` acts on ``(i, j)``.
"""
from __future__ import annotations

import collections
import time

import numpy as np

from ..bb import bb_code
from ..core import TannerGraph
from ..lp_sandwich import _monomial_index
from ..schedule import Schedule
from .build import sector
from .bounds import upper_bposd
from .residual import extended_matrices

__all__ = ["Tiling", "cegar", "linear_colouring", "random_colouring"]

KINDS = ("P0", "P4", "T3")          # first pair, last pair, last triple (ranks)


def linear_colouring(l, m, forms):
    """``(ty, i, j) -> tuple((si*i + sj*j) % mod for each form)``; the same
    for both check types.  ``forms=[(1, 1, 3)]`` is Strikis et al.'s
    ``i+j mod 3``; ``[(1, 1, 3), (1, 0, 2)]`` refines it to six classes."""
    def phi(ty, i, j):
        return tuple((si * i + sj * j) % mod for si, sj, mod in forms)
    return phi


def random_colouring(tiling, k, seed=0, tl=30.0):
    """A random proper ``k``-colouring of each check type's overlap graph
    (X and Z independently), by CP-SAT with a random objective so that it is
    not translation invariant.  Returns ``{'X': {cid: c}, 'Z': {...}}``."""
    from ortools.sat.python import cp_model

    rng = np.random.default_rng(seed)
    out = {}
    for ty in ("X", "Z"):
        md = cp_model.CpModel()
        cids = sorted(tiling.nbrs[ty])
        col = {cid: md.NewIntVar(0, k - 1, "") for cid in cids}
        for a, b in tiling.check_overlaps[ty]:
            md.Add(col[a] != col[b])
        md.Add(col[cids[0]] == 0)
        md.Maximize(sum(int(rng.integers(-5, 6)) * col[cid] for cid in cids))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = tl
        s.parameters.num_search_workers = 4
        s.parameters.random_seed = seed
        st = s.Solve(md)
        assert st in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"no proper {k}-colouring found"
        out[ty] = {cid: int(s.Value(col[cid])) for cid in cids}
    return out


class Tiling:
    """A BB code with edge types, a colouring of its checks and the tiled model."""

    def __init__(self, l, m, a_terms, b_terms, colouring=None, name=""):
        self.l, self.m = l, m
        self.code = bb_code(l, m, a_terms, b_terms, name=name)
        self.graph = TannerGraph.build(self.code)
        self.n, self.nblk = self.code.n, l * m
        ia, ib = _monomial_index(l, m, a_terms, self.nblk), _monomial_index(l, m, b_terms, self.nblk)
        E = len(self.graph.edges)
        self.typ = np.zeros(E, dtype=int)          # 0 = X edge, 1 = Z edge
        self.lab = np.zeros(E, dtype=int)          # type 0..5
        self.cid = np.zeros(E, dtype=int)
        self.qub = np.zeros(E, dtype=int)
        self.nbrs = {"X": collections.defaultdict(list), "Z": collections.defaultdict(list)}
        for e, (ck, q, ty) in enumerate(self.graph.edges):
            cid, left = ck[1], q < self.nblk
            qq = q if left else q - self.nblk
            if ty == "X":
                t = int(ia[cid, qq]) if left else 3 + int(ib[cid, qq])
            else:
                self.typ[e] = 1
                t = 3 + int(ib[qq, cid]) if left else int(ia[qq, cid])
            self.lab[e], self.cid[e], self.qub[e] = t, cid, q
            self.nbrs[ty][cid].append((t, q))
        for ty in ("X", "Z"):
            for cid, lst in self.nbrs[ty].items():
                assert sorted(t for t, _q in lst) == list(range(6)), (ty, cid, lst)
        # same-type overlap graphs (for colourings) and X/Z overlaps (for properness)
        self.check_overlaps = {}
        for ty in ("X", "Z"):
            byq = collections.defaultdict(list)
            for cid, lst in self.nbrs[ty].items():
                for _t, q in lst:
                    byq[q].append(cid)
            pairs = set()
            for cs in byq.values():
                for a in cs:
                    for b in cs:
                        if a < b:
                            pairs.add((a, b))
            self.check_overlaps[ty] = sorted(pairs)
        zby = collections.defaultdict(list)
        for cid, lst in self.nbrs["Z"].items():
            for t, q in lst:
                zby[q].append((cid, t))
        self.overlaps = {}                          # X cid -> {Z cid: [(tX, tZ), ...]}
        for g, lst in self.nbrs["X"].items():
            d = collections.defaultdict(list)
            for tX, q in lst:
                for h, tZ in zby[q]:
                    d[h].append((tX, tZ))
            self.overlaps[g] = dict(d)
        self.set_colouring(colouring if colouring is not None else linear_colouring(l, m, [(1, 1, 3)]))

    # ---------------------------------------------------------------- colouring
    def set_colouring(self, colouring):
        """``colouring``: callable ``(ty, i, j) -> class`` or ``{'X': {cid: c}, 'Z': {...}}``."""
        m = self.m
        if callable(colouring):
            self.colour = {ty: {cid: colouring(ty, cid // m, cid % m) for cid in self.nbrs[ty]}
                           for ty in ("X", "Z")}
        else:
            self.colour = {ty: dict(colouring[ty]) for ty in ("X", "Z")}
        for ty in ("X", "Z"):
            for a, b in self.check_overlaps[ty]:
                assert self.colour[ty][a] != self.colour[ty][b], \
                    f"colouring is not proper on the {ty} checks ({a}, {b})"
        self.classes = {ty: sorted(set(self.colour[ty].values())) for ty in ("X", "Z")}

    def translate(self, cid, d):
        i, j = cid // self.m, cid % self.m
        return ((i + d[0]) % self.l) * self.m + (j + d[1]) % self.m

    # ------------------------------------------------------------------ circuits
    def times_of(self, TX, TZ):
        """Edge layers from per-class type layers ``TX[c][t]``, ``TZ[c][t]``."""
        tt = np.zeros(len(self.lab), dtype=int)
        for e in range(len(self.lab)):
            ty = "X" if self.typ[e] == 0 else "Z"
            c = self.colour[ty][self.cid[e]]
            tt[e] = (TX if ty == "X" else TZ)[c][self.lab[e]]
        return tt

    def residual_columns(self, side, TX, TZ):
        """Residual columns of the hook checks bounding ``side`` (``'Z'`` basis
        <- X-check residuals).  Returns ``[(support, cid, kind, typeset), ...]``."""
        hook = "X" if side == "Z" else "Z"
        T = TX if hook == "X" else TZ
        cols = []
        for cid, lst in self.nbrs[hook].items():
            q_of = dict(lst)
            order = sorted(range(6), key=lambda t: T[self.colour[hook][cid]][t])
            for kind, types in (("P0", order[:2]), ("T3", order[3:]), ("P4", order[4:])):
                sup = frozenset(q_of[t] for t in types)
                cols.append((sup, cid, kind, tuple(sorted(types))))
        return cols

    def verify(self, TX, TZ, D):
        s = Schedule(code=self.code, graph=self.graph, times=self.times_of(TX, TZ), T=D)
        return s.verify()

    # ------------------------------------------------------------- reduced model
    def reduced(self, D, e, nogoods, *, seed=0, tl=60.0, workers=8, forbid=(), hint=None,
                sandwich=True):
        """CP-SAT: a proper sandwich-band tiled circuit of depth ``D`` with the
        middle band ``[e, e+3)`` avoiding every no-good.

        A no-good is a frozenset of ``(hook, class, kind, typeset)`` literals:
        the constraint says they are not all present.  ``forbid`` lists exact
        ``(TX, TZ)`` tilings to exclude (previous survivors).
        Returns ``(status, TX, TZ)`` with ``TX[c][t]`` dicts of lists.
        """
        from ortools.sat.python import cp_model

        md = cp_model.CpModel()
        T = {"X": {}, "Z": {}}
        for ty in ("X", "Z"):
            for c in self.classes[ty]:
                for t in range(6):
                    T[ty][(c, t)] = md.NewIntVar(0, D - 1, "")
                md.AddAllDifferent([T[ty][(c, t)] for t in range(6)])
                if sandwich:
                    for t in range(6):
                        v = T[ty][(c, t)]
                        if t < 3:                        # a / a* : middle band
                            md.Add(v >= e)
                            md.Add(v <= e + 2)
                        else:                            # b / b* : outside it
                            lo = md.NewBoolVar("")
                            md.Add(v <= e - 1).OnlyEnforceIf(lo)
                            md.Add(v >= e + 3).OnlyEnforceIf(lo.Not())
        # qubit collisions
        inc = collections.defaultdict(set)
        for e_ in range(len(self.lab)):
            ty = "X" if self.typ[e_] == 0 else "Z"
            inc[self.qub[e_]].add((ty, self.colour[ty][self.cid[e_]], self.lab[e_]))
        for key in {frozenset(s) for s in inc.values()}:
            md.AddAllDifferent([T[ty][(c, t)] for ty, c, t in key])
        # properness on every overlapping pair
        seen = set()
        for g, d in self.overlaps.items():
            for h, ps in d.items():
                key = (self.colour["X"][g], self.colour["Z"][h], tuple(sorted(ps)))
                if key in seen:
                    continue
                seen.add(key)
                cX, cZ, ps = key
                bs = []
                for tX, tZ in ps:
                    b = md.NewBoolVar("")
                    md.Add(T["X"][(cX, tX)] < T["Z"][(cZ, tZ)]).OnlyEnforceIf(b)
                    md.Add(T["X"][(cX, tX)] > T["Z"][(cZ, tZ)]).OnlyEnforceIf(b.Not())
                    bs.append(b)
                md.AddBoolXOr(bs + [md.NewConstant(1)])   # even number of "x before z"
        o, rk, pres = {}, {}, {}

        def before(ty, c, t, u):
            k = (ty, c, t, u)
            if k not in o:
                b = md.NewBoolVar("")
                md.Add(T[ty][(c, t)] < T[ty][(c, u)]).OnlyEnforceIf(b)
                md.Add(T[ty][(c, t)] > T[ty][(c, u)]).OnlyEnforceIf(b.Not())
                o[k] = b
            return o[k]

        def rank_le(ty, c, t, r):
            k = (ty, c, t, r)
            if k not in rk:
                b = md.NewBoolVar("")
                s = sum(before(ty, c, u, t) for u in range(6) if u != t)
                md.Add(s <= r).OnlyEnforceIf(b)
                md.Add(s >= r + 1).OnlyEnforceIf(b.Not())
                rk[k] = b
            return rk[k]

        def present(hook, c, kind, types):
            k = (hook, c, kind, types)
            if k not in pres:
                if kind == "P0":
                    lits = [rank_le(hook, c, t, 1) for t in types]
                elif kind == "P4":
                    lits = [rank_le(hook, c, t, 3).Not() for t in types]
                else:
                    lits = [rank_le(hook, c, t, 2).Not() for t in types]
                b = md.NewBoolVar("")
                md.AddBoolAnd(lits).OnlyEnforceIf(b)
                md.AddBoolOr([x.Not() for x in lits]).OnlyEnforceIf(b.Not())
                pres[k] = b
            return pres[k]

        for ng in nogoods:
            md.AddBoolOr([present(hook, c, kind, types).Not() for hook, c, kind, types in ng])
        for TXf, TZf in forbid:
            diff = []
            for ty, Tf in (("X", TXf), ("Z", TZf)):
                for c in self.classes[ty]:
                    for t in range(6):
                        b = md.NewBoolVar("")
                        md.Add(T[ty][(c, t)] != Tf[c][t]).OnlyEnforceIf(b)
                        md.Add(T[ty][(c, t)] == Tf[c][t]).OnlyEnforceIf(b.Not())
                        diff.append(b)
            md.AddBoolOr(diff)
        if hint is not None:
            for ty, Th in (("X", hint[0]), ("Z", hint[1])):
                for c in self.classes[ty]:
                    for t in range(6):
                        md.AddHint(T[ty][(c, t)], int(Th[c][t]))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = tl
        s.parameters.num_search_workers = workers
        s.parameters.random_seed = seed
        st = s.Solve(md)
        name = s.StatusName(st)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            TX = {c: [int(s.Value(T["X"][(c, t)])) for t in range(6)] for c in self.classes["X"]}
            TZ = {c: [int(s.Value(T["Z"][(c, t)])) for t in range(6)] for c in self.classes["Z"]}
            return name, TX, TZ
        return name, None, None

    # ---------------------------------------------------------------- motifs
    def extended(self, side, cols):
        sec = sector(self.code, side)
        seen, R, keep = set(), [], []
        for sup, cid, kind, types in cols:
            if sup in seen:
                continue
            seen.add(sup)
            v = np.zeros(self.n, dtype=np.uint8)
            v[list(sup)] = 1
            R.append(v)
            keep.append((sup, cid, kind, types))
        R = np.array(R, dtype=np.uint8).T
        He, Le = extended_matrices(sec.H, sec.L, R)
        return He, Le, keep

    def find_motif(self, side, cols, *, wmax=10, workers=8, tl_first=120.0, tl_min=60.0,
                   bposd_attempts=40):
        """A logical of the extended code of weight ``<= wmax`` using as few
        residual columns as CP-SAT can manage in ``tl_min`` seconds.

        Stage 1: BP-OSD, then CP-SAT first-witness (``tl_first``) to decide
        existence cheaply.  Stage 2: minimise the residual-column count under
        the cap, warm-started from the witness.  Returns ``(columns, weight,
        how)`` or ``None`` (no witness found: NOT a proof of absence).
        """
        from ortools.sat.python import cp_model

        He, Le, keep = self.extended(side, cols)
        n = self.n
        wit = None
        est = upper_bposd(He, Le, attempts=bposd_attempts, basis=side, quantity="d_ext")
        if est is not None and est.witness is not None and int(est.value) <= wmax:
            w = np.asarray(est.witness, dtype=np.uint8)
            if not ((He @ w) % 2).any() and ((Le @ w) % 2).any():
                wit = w

        def model(minimise):
            md = cp_model.CpModel()
            x = [md.NewBoolVar(f"x{i}") for i in range(He.shape[1])]
            TRUE = md.NewConstant(1)
            for row in He:
                sup = np.nonzero(row)[0].tolist()
                if sup:
                    md.AddBoolXOr([x[i] for i in sup] + [TRUE])
            bits = []
            for row in Le:
                sup = np.nonzero(row)[0].tolist()
                if not sup:
                    continue
                b = md.NewBoolVar("")
                md.AddBoolXOr([x[i] for i in sup] + [b.Not()])
                bits.append(b)
            md.Add(sum(bits) >= 1)
            md.Add(sum(x) <= wmax)
            if minimise:
                md.Minimize(sum(x[n:]))
            return md, x

        if wit is None:
            md, x = model(False)
            s = cp_model.CpSolver()
            s.parameters.max_time_in_seconds = tl_first
            s.parameters.num_search_workers = workers

            class _Stop(cp_model.CpSolverSolutionCallback):
                def on_solution_callback(self):
                    self.StopSearch()
            st = s.Solve(md, _Stop())
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None
            wit = np.array([int(s.Value(v)) for v in x], dtype=np.uint8)
            how = "cpsat"
        else:
            how = "bposd"
        md, x = model(True)
        for i in range(len(x)):
            md.AddHint(x[i], int(wit[i]))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = tl_min
        s.parameters.num_search_workers = workers
        st = s.Solve(md)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            wit = np.array([int(s.Value(v)) for v in x], dtype=np.uint8)
            how += "+min" + ("*" if st == cp_model.OPTIMAL else "")
        assert not ((He @ wit) % 2).any() and ((Le @ wit) % 2).any() and wit.sum() <= wmax
        used = [keep[i - n] for i in np.nonzero(wit[n:])[0]]
        return used, int(wit.sum()), how

    def nogoods_from(self, side, used):
        """The no-good of a motif at every translate of the code (deduplicated).

        Sound for ANY colouring: if the translated columns were all present,
        translating the logical back would give one on the original columns."""
        hook = "X" if side == "Z" else "Z"
        out = set()
        for di in range(self.l):
            for dj in range(self.m):
                out.add(frozenset((hook, self.colour[hook][self.translate(cid, (di, dj))], kind, types)
                                  for _sup, cid, kind, types in used))
        return sorted(out, key=lambda ng: sorted(map(str, ng)))


def cegar(tiling: Tiling, D: int, e: int, *, max_iter=200, seed=0, workers=8,
          tl_reduced=60.0, tl_first=120.0, tl_min=60.0, wmax=10, survivors_wanted=3,
          sandwich=True, log=print, nogoods=None):
    """The loop.  Returns ``(survivors, nogoods, status)``; ``status`` is
    ``'INFEASIBLE'`` (certificate for the family), ``'survivors'`` or ``'budget'``."""
    nogoods = list(nogoods or [])
    survivors: list = []
    forbid: list = []
    t0 = time.time()
    hint = None
    for it in range(max_iter):
        st, TX, TZ = tiling.reduced(D, e, nogoods, seed=seed + it, tl=tl_reduced, workers=workers,
                                    forbid=forbid, hint=hint, sandwich=sandwich)
        if TX is None:
            log(f"[{it}] reduced model {st} with {len(nogoods)} no-goods ({time.time()-t0:.0f}s)")
            return survivors, nogoods, st
        rep = tiling.verify(TX, TZ, D)
        assert rep["proper"] and rep["contam"] and rep["sim"] and rep["depth"] <= D, rep
        killed = False
        for side in ("Z", "X"):
            cols = tiling.residual_columns(side, TX, TZ)
            t = time.time()
            r = tiling.find_motif(side, cols, wmax=wmax, workers=workers, tl_first=tl_first,
                                  tl_min=tl_min)
            if r is None:
                log(f"[{it}] {side}: no logical <= {wmax} found ({time.time()-t:.0f}s)")
                continue
            used, w, how = r
            new = tiling.nogoods_from(side, used)
            before = len(nogoods)
            have = set(nogoods)
            for ng in new:
                if ng not in have:
                    nogoods.append(ng)
                    have.add(ng)
            log(f"[{it}] {side}: weight-{w} logical on {len(used)} residual columns "
                f"({how}, {time.time()-t:.0f}s) -> +{len(nogoods)-before} = {len(nogoods)} no-goods")
            killed = True
            break
        if not killed:
            survivors.append((TX, TZ))
            forbid.append((TX, TZ))
            log(f"[{it}] SURVIVOR #{len(survivors)}: TX={TX} TZ={TZ}")
            if len(survivors) >= survivors_wanted:
                return survivors, nogoods, "survivors"
        hint = (TX, TZ)
    return survivors, nogoods, "budget"
