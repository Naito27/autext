"""Bounds on the circuit distance, each carrying its own direction.

For a noisy circuit with detector matrix ``D`` and observable matrix ``O``,

    d_circ = min { |F| : D F = 0,  O F != 0 } ,

the fewest faults that go undetected yet flip a logical. Three facts govern
everything in this module, and they are why a bound here is a typed
:class:`Bound` and not an int:

1. ``d_circ`` is a MINIMUM. A witness (a fault set that works) is an UPPER
   bound; only a refutation (nothing lighter exists) is a LOWER bound.
2. Strikis, Browne and Beverland (arXiv:2603.05481) Thm 1: the non-interleaved
   counterpart has ``d'_circ = d_ext``, the extended-code distance over the
   FULL residual set, and ``d'_circ <= d_circ``. So an *exact* ``d_ext`` is a
   lower bound on ``d_circ``; a mere witness for ``d_ext`` bounds nothing.
3. ``d_circ`` is the min over the two memory bases, so a lower bound on the
   total needs a lower bound in EVERY basis.

A :class:`Bound` states one guaranteed inequality about one quantity. There is
no ``proved`` flag to misread: a solver that could not prove returns the weaker
kind, or ``None``. :func:`bracket` combines bounds and reports an exact value
only when a lower and an upper bound meet.

Solvers, cheapest first:

- :func:`upper_stim`    stim's own search; exact once the explored class covers
  every mechanism, i.e. once the limit reaches the largest detector degree.
- :func:`upper_bposd`   Strikis et al.'s adaptive BP-OSD.
- :func:`lower_ext`     exact ``d_ext`` by CP-SAT on the extended code; the
  matrix is ``n + |residuals|`` wide against the DEM's mechanism count.
- :func:`exact_cpsat`   CP-SAT on the DEM, XOR-encoded, optionally with the
  orbit cut of :mod:`.symmetry`. By far the most expensive route, and the only
  one that can prove an exact value on the DEM itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .build import dem, sector

__all__ = ["Bound", "Bracket", "bracket", "min_over_bases", "upper_stim",
           "upper_bposd", "ext_distance", "lower_ext", "exact_cpsat"]


@dataclass(frozen=True)
class Bound:
    """One guaranteed inequality: ``quantity`` ``<=`` / ``>=`` / ``==`` ``value``."""

    value: int
    kind: str                 # 'upper' | 'lower' | 'exact'
    quantity: str = "d_circ"  # or 'd_ext'
    method: str = ""
    seconds: float = 0.0
    basis: str | None = None
    witness: object = field(default=None, compare=False, repr=False)  # the fault set, if any

    def __post_init__(self):
        if self.kind not in ("upper", "lower", "exact"):
            raise ValueError(f"kind must be upper/lower/exact, got {self.kind!r}")
        if self.value is None:
            raise ValueError("a Bound with no value is not a bound; return None")

    @property
    def lo(self):
        return self.value if self.kind in ("lower", "exact") else None

    @property
    def hi(self):
        return self.value if self.kind in ("upper", "exact") else None

    def __str__(self):
        sym = {"upper": "<=", "lower": ">=", "exact": "="}[self.kind]
        b = f"({self.basis})" if self.basis else ""
        return f"{self.quantity}{b} {sym} {self.value} [{self.method}, {self.seconds:.1f}s]"


@dataclass(frozen=True)
class Bracket:
    lo: int | None
    hi: int | None
    quantity: str = "d_circ"
    basis: str | None = None

    @property
    def exact(self):
        return self.lo if (self.lo is not None and self.lo == self.hi) else None

    def __str__(self):
        b = f"({self.basis})" if self.basis else ""
        if self.exact is not None:
            return f"{self.quantity}{b} = {self.exact}"
        lo = self.lo if self.lo is not None else "?"
        hi = self.hi if self.hi is not None else "?"
        return f"{lo} <= {self.quantity}{b} <= {hi}"


def bracket(*bounds, quantity="d_circ", basis=None) -> Bracket:
    """Tightest interval implied by the given bounds on one quantity/basis.

    Raises if a lower bound exceeds an upper bound: that means a solver is
    wrong, and it must not be silently reconciled.
    """
    los = [b.lo for b in bounds if b is not None and b.quantity == quantity
           and b.basis == basis and b.lo is not None]
    his = [b.hi for b in bounds if b is not None and b.quantity == quantity
           and b.basis == basis and b.hi is not None]
    lo = max(los) if los else None
    hi = min(his) if his else None
    if lo is not None and hi is not None and lo > hi:
        raise AssertionError(f"inconsistent bounds: lo={lo} > hi={hi} -- a solver is wrong")
    return Bracket(lo, hi, quantity, basis)


def min_over_bases(z: Bracket, x: Bracket) -> Bracket:
    """``d_circ = min(d_circ(Z), d_circ(X))``.

    The upper bound is the min of the uppers. The LOWER bound needs a lower
    bound in BOTH bases --- with one missing, the total is unbounded below.
    """
    his = [v for v in (z.hi, x.hi) if v is not None]
    hi = min(his) if his else None
    lo = min(z.lo, x.lo) if (z.lo is not None and x.lo is not None) else None
    return Bracket(lo, hi, "d_circ", None)


# --------------------------------------------------------------------- upper
def upper_stim(code, times, graph, basis: str, lim: int = 4, p: float = 1e-3,
               deg_lim=None, no_increase: bool = False):
    """stim's ``search_for_undetectable_logical_errors``: an UPPER bound.

    It is the minimum over the mechanisms it explores (those flipping at most
    ``lim`` detectors, ``deg_lim`` per edge --- default ``lim`` --- and, with
    ``no_increase``, never raising the symptom degree), so a genuine upper
    bound always, and exact once the limits cover every mechanism. ``None`` if
    nothing was found within the class. Cost grows steeply with ``lim``.
    """
    t = time.time()
    deg_lim = lim if deg_lim is None else deg_lim
    circ, _D, _O = dem(code, times, graph, basis, p=p)
    try:
        errs = circ.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=lim,
            dont_explore_edges_with_degree_above=deg_lim,
            dont_explore_edges_increasing_symptom_degree=no_increase)
    except Exception:
        return None
    tag = f"stim lim={lim}" + (f"/{deg_lim}" if deg_lim != lim else "") + ("/noinc" if no_increase else "")
    return Bound(len(errs), "upper", "d_circ", tag, time.time() - t, basis)


def _bposd_min_weight(H, L, attempts, seed, error_rate, max_iter, osd_order):
    """Min weight ``x`` with ``H x = 0``, ``L x != 0``, after Strikis et al.'s
    ``BPOSD_ext_dist``.

    The constraint row is a RANDOM nontrivial logical (random combination of
    ``L`` rows plus random combination of ``H`` rows --- the latter is invisible
    to any admissible ``x`` since ``H x = 0``), and the attempt counter resets
    on every improvement. Randomising over logicals matters: constraining a
    single fixed ``L`` row only ever overstates the minimum weight.
    """
    from ldpc import BpOsdDecoder

    E = H.shape[1]
    if not L.any():
        return None
    rng = np.random.default_rng(seed)
    best, wit, miss = E, None, 0
    while miss < attempts:
        v = rng.integers(0, 2, size=L.shape[0])
        while not v.any():
            v = rng.integers(0, 2, size=L.shape[0])
        row = ((v @ L) + (rng.integers(0, 2, size=H.shape[0]) @ H)) % 2
        Hs = np.vstack([H, row[None, :]]).astype(np.uint8)
        syn = np.zeros(Hs.shape[0], dtype=np.uint8)
        syn[-1] = 1
        dec = BpOsdDecoder(Hs, error_rate=error_rate, max_iter=max_iter,
                           bp_method="ms", ms_scaling_factor=0.625,
                           osd_method="osd_cs", osd_order=osd_order)
        x = dec.decode(syn)
        wt = int(np.count_nonzero(x))
        if wt and not ((Hs @ x) % 2 != syn).any() and L.dot(x).any():
            if wt < best:
                best, wit, miss = wt, np.asarray(x, dtype=np.uint8).copy(), 0
                continue
        miss += 1
    return (int(best), wit) if best < E else (None, None)


def upper_bposd(D, O, attempts: int = 30, seed: int = 0, error_rate: float = 0.05,
                max_iter: int = 30, osd_order: int = 10, basis=None,
                quantity: str = "d_circ"):
    """Adaptive BP-OSD (Strikis et al.) on ``(D, O)``: an UPPER bound, or
    ``None``."""
    t = time.time()
    w, wit = _bposd_min_weight(D.astype(np.uint8), O.astype(np.uint8), attempts, seed,
                               error_rate, max_iter, osd_order)
    if w is None:
        return None
    return Bound(w, "upper", quantity, "bposd", time.time() - t, basis, witness=wit)


# --------------------------------------------------------------------- exact / lower
def _cpsat_min_weight(H, L, time_limit, workers, ub=None, cut=None, hint=None,
                      first=False):
    """``(status, value)``, status in OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN.

    XOR-encoded (CP-SAT's propagator does Gaussian elimination), minimised
    under a cap (minimising keeps the incumbent-based pruning that pure
    feasibility loses), with an optional symmetry cut
    ``sum_{m in cut} x_m >= 1``.
    ``first`` stops at the FIRST solution: with a cap that is a killer test
    ("is there anything <= ub?") whose answer is FEASIBLE the moment a
    witness appears, instead of minimising on to the time limit.
    """
    from ortools.sat.python import cp_model

    N = H.shape[1]
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(N)]
    TRUE = m.NewConstant(1)
    for row in H:
        sup = np.nonzero(row)[0].tolist()
        if sup:
            m.AddBoolXOr([x[i] for i in sup] + [TRUE])
    bits = []
    for row in L:
        sup = np.nonzero(row)[0].tolist()
        if not sup:
            continue
        b = m.NewBoolVar("")
        m.AddBoolXOr([x[i] for i in sup] + [b.Not()])
        bits.append(b)
    if not bits:
        return "INFEASIBLE", None
    m.Add(sum(bits) >= 1)
    if cut:
        m.Add(sum(x[i] for i in cut) >= 1)
    if ub is not None:
        m.Add(sum(x) <= int(ub))
    m.Minimize(sum(x))
    if hint is not None:                 # warm start: an incumbent to refute below
        for i in range(N):
            m.AddHint(x[i], int(hint[i]))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    if first:
        class _Stop(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                self.StopSearch()
        st = s.Solve(m, _Stop())
    else:
        st = s.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        xv = np.array([int(s.Value(v)) for v in x], dtype=np.uint8)
        val = int(xv.sum())
        if first and st == cp_model.OPTIMAL and val > 0:
            st = cp_model.FEASIBLE          # a stopped search proves nothing optimal
        return ("OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE", val, xv)
    if st == cp_model.INFEASIBLE:
        return "INFEASIBLE", None, None
    return "UNKNOWN", None, None


def _status_to_bound(status, value, ub, quantity, method, secs, basis, witness=None):
    if status == "OPTIMAL":
        return Bound(value, "exact", quantity, method, secs, basis, witness=witness)
    if status == "FEASIBLE":
        return Bound(value, "upper", quantity, method + " (witness only)", secs, basis,
                     witness=witness)
    if status == "INFEASIBLE" and ub is not None:
        return Bound(int(ub) + 1, "lower", quantity, method + f" (refuted <= {ub})",
                     secs, basis)
    return None


def exact_cpsat(D, O, time_limit: float = 300.0, workers: int = 8, ub=None,
                cut=None, basis=None, hint=None, first: bool = False):
    """CP-SAT on the DEM. Returns the strongest :class:`Bound` it can justify.

    ``ub`` caps the weight (a stim/BP-OSD witness, or the code distance, which
    always bounds ``d_circ`` above). ``cut`` is a list of orbit representatives
    from :mod:`.symmetry`; valid because any solution translates to contain one.
    OPTIMAL -> exact; FEASIBLE -> upper; INFEASIBLE under the cap -> lower.
    """
    t = time.time()
    st, val, xv = _cpsat_min_weight(D, O, time_limit, workers, ub=ub, cut=cut, hint=hint,
                                    first=first)
    return _status_to_bound(st, val, ub, "d_circ", "cpsat", time.time() - t, basis, xv)


def ext_distance(H_ext, L_ext, time_limit: float = 300.0, workers: int = 8,
                 ub=None, basis=None, first: bool = False):
    """``d_ext`` of an extended code: a :class:`Bound` on quantity ``'d_ext'``.

    ``first`` with a cap ``ub`` is the killer test: return at the first
    witness of weight ``<= ub`` (an upper bound) instead of minimising on."""
    t = time.time()
    st, val, xv = _cpsat_min_weight(H_ext, L_ext, time_limit, workers, ub=ub, first=first)
    return _status_to_bound(st, val, ub, "d_ext", "cpsat", time.time() - t, basis, xv)


def lower_ext(code, times, graph, basis: str, time_limit: float = 300.0,
              workers: int = 8):
    """Strikis et al. Thm 1 as a LOWER bound on ``d_circ`` in one basis.

    Builds the extended code over the FULL residual set of the schedule and
    solves it exactly. Only an exact ``d_ext`` transfers: it equals ``d'_circ``,
    the non-interleaved counterpart's distance, which is ``<= d_circ``. A mere
    witness for ``d_ext`` is returned as an upper bound on ``d_ext`` and says
    nothing about ``d_circ`` --- callers wanting the circuit bound should use
    :func:`bracket` on quantity ``'d_circ'`` and will simply not see it.
    """
    from .residual import extended_matrices, tail_matrix, tails

    sec = sector(code, basis)
    R = tail_matrix(code, tails(code, times, graph, basis))
    He, Le = extended_matrices(sec.H, sec.L, R)
    b = ext_distance(He, Le, time_limit=time_limit, workers=workers, basis=basis)
    if b is None or b.kind != "exact":
        return b
    return Bound(b.value, "lower", "d_circ", "ext (Thm 1)", b.seconds, basis,
                 witness=b.witness)
