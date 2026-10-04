"""One noisy round of syndrome extraction, as a stim circuit and as matrices.

Everything downstream --- every bound in :mod:`.bounds`, every residual in
:mod:`.residual` --- starts from the objects built here:

- :func:`memory_circuit`  a noisy ``basis``-basis memory experiment,
- :func:`dem_matrices`    its detector error model as ``(D, O)``,
- :func:`deinterleave`    the non-interleaved counterpart of a schedule, after
  Strikis, Browne and Beverland (arXiv:2603.05481),
- :func:`sector`          **the one place** that says, for a memory basis, which
  parity checks detect the errors that matter, which logicals are observed, and
  which checks' ancilla faults produce hooks.

**Convention: one noisy round, perfect final readout.** The companion paper's
conclusion argues that time-reversing every second round makes the circuit
distance of a full memory experiment equal that of a single round. ``rounds``
is exposed, but the convention used throughout is ``rounds=1``.

**Noise** (Strikis et al. Sec. II; also Zhang et al., arXiv:2603.21499): every
operation fails independently with probability ``p`` --- two-qubit depolarizing
after each CX, one-qubit depolarizing on each idle qubit in each layer, a flip
on each reset and on each measurement. ``d_circ`` counts fault locations, so it
does not depend on ``p``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import CSSCode, TannerGraph
from ..distance import _logical_basis

__all__ = ["Sector", "sector", "logical_basis", "memory_circuit", "deinterleave",
           "dem_matrices", "drop_dead", "dem"]


@dataclass(frozen=True)
class Sector:
    """What a memory basis looks at.

    ``basis='Z'``: data start in ``|0>``, the observables are Z-type logicals,
    and they are flipped by **X** errors. X errors are detected by the rows of
    ``H_Z``; an X error is spread onto data by a fault on an **X-ancilla**
    (its CNOTs run ancilla-control to data-target, so an X on the control
    lands on every data qubit the check has yet to touch), while a fault on a
    Z-ancilla propagates the other way and never reaches the data. Hence for
    the Z basis: ``H = H_Z``, ``L = Z-logicals``, ``hook_checks = 'X'``.
    The X basis is the mirror image.
    """

    basis: str
    H: np.ndarray          # detects the error type that flips the observable
    L: np.ndarray          # the observed logicals, one per row
    hook_checks: str       # which check family's ancilla faults become hooks


def logical_basis(code: CSSCode, basis: str):
    """Rows spanning the logicals observed in ``basis``-basis memory."""
    if basis == "Z":
        return _logical_basis(code.HZ, code.HX, code.k)   # ker H_X mod rowspace H_Z
    if basis == "X":
        return _logical_basis(code.HX, code.HZ, code.k)
    raise ValueError("basis must be 'X' or 'Z'")


def sector(code: CSSCode, basis: str) -> Sector:
    L = np.array(logical_basis(code, basis), dtype=np.uint8)
    if basis == "Z":
        return Sector("Z", code.HZ.astype(np.uint8), L, "X")
    return Sector("X", code.HX.astype(np.uint8), L, "Z")


def _gate_layers(code: CSSCode, times, graph: TannerGraph):
    """``layers[t]`` = list of ``(control, target)`` CX pairs in layer ``t``."""
    n, rX = code.n, code.rx
    T = int(np.max(times)) + 1
    layers = [[] for _ in range(T)]
    for e, (ck, q, ty) in enumerate(graph.edges):
        t = int(times[e])
        if ty == "X":
            layers[t].append((n + ck[1], q))          # X-ancilla controls data
        else:
            layers[t].append((q, n + rX + ck[1]))     # data controls Z-ancilla
    return layers


def memory_circuit(code: CSSCode, times, graph: TannerGraph | None = None, *,
                   basis: str = "Z", p: float = 1e-3, idle: bool = True,
                   rounds: int = 1, round_times=None, readout_noise: bool = False,
                   meas_idle: bool = False, ideal_prep: bool = False):
    """Noisy ``basis``-basis memory: ``rounds`` noisy rounds, perfect readout.

    Detector layout is family-major and is relied on by :mod:`.symmetry`:
    round ``r``'s checks occupy ``r*r_check ...``, then the final block that
    compares the last round against the data readout.

    ``meas_idle`` adds the idling channel on the DATA while the ancillas are
    measured and reset.  Bravyi et al. (arXiv:2308.07915) Table 5 lists exactly
    that ("Idle q(L), Idle q(R)" in its round 8), and leaving it out drops one
    fault location per data qubit per round.  ``ideal_prep`` makes the initial
    logical state preparation noiseless, the convention in which only the QEC
    cycle is noisy.

    ``readout_noise`` makes the FINAL data readout noisy too.  The default is
    the one-noisy-round convention with perfect final readout, but that
    leaves the two temporal boundaries inequivalent -- the data preparation is
    noisy and the readout is not -- so time reversal of the CNOT order need not
    preserve ``d_circ``.  With it on, every preparation is matched to a
    measurement in the same basis and the spacetime code is time-symmetric.

    ``round_times`` gives a DIFFERENT schedule per round (a list of ``rounds``
    layer arrays, overriding ``times``). Alternating a schedule with its time
    reverse is Bravyi et al.'s open question 4: a residual that one CNOT order
    cannot detect is a tail of that order, and the reversed order's tails are
    the complementary prefixes, so two circuits in tandem may catch fault
    paths neither catches alone.
    """
    import stim

    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    graph = graph if graph is not None else TannerGraph.build(code)
    n, rX, rZ = code.n, code.rx, code.rz
    data = list(range(n))
    xanc = list(range(n, n + rX))
    zanc = list(range(n + rX, n + rX + rZ))
    if round_times is None:
        per_round = [_gate_layers(code, times, graph)] * rounds
    else:
        per_round = [_gate_layers(code, np.asarray(t), graph) for t in round_times]
        if len(per_round) != rounds:
            raise ValueError(f"round_times has {len(per_round)} schedules for {rounds} rounds")

    c = stim.Circuit()
    nmeas = 0
    z_rec: list[dict[int, int]] = []
    x_rec: list[dict[int, int]] = []

    c.append("RX" if basis == "X" else "R", data)
    c.append("RX", xanc)
    c.append("R", zanc)
    if p:
        if not ideal_prep:
            c.append("Z_ERROR" if basis == "X" else "X_ERROR", data, p)
        c.append("Z_ERROR", xanc, p)
        c.append("X_ERROR", zanc, p)

    for r in range(rounds):
        for pairs in per_round[r]:
            flat = [q for pair in pairs for q in pair]
            c.append("CX", flat)
            if p:
                c.append("DEPOLARIZE2", flat, p)
                if idle:
                    busy = set(flat)
                    rest = [q for q in range(n + rX + rZ) if q not in busy]
                    if rest:
                        c.append("DEPOLARIZE1", rest, p)
            c.append("TICK")
        if p and meas_idle and idle:
            c.append("DEPOLARIZE1", data, p)      # data idle while ancillas are read out
        c.append("MX", xanc, p)
        x_rec.append({a: nmeas + a for a in range(rX)})
        nmeas += rX
        c.append("M", zanc, p)
        z_rec.append({b: nmeas + b for b in range(rZ)})
        nmeas += rZ
        if r + 1 < rounds:
            c.append("RX", xanc)
            c.append("R", zanc)
            if p:
                c.append("Z_ERROR", xanc, p)
                c.append("X_ERROR", zanc, p)

    c.append("MX" if basis == "X" else "M", data, p if readout_noise else 0)
    d_rec = {q: nmeas + q for q in range(n)}
    nmeas += n

    def rec(i):
        return stim.target_rec(i - nmeas)

    checks, anc_rec = (code.HZ, z_rec) if basis == "Z" else (code.HX, x_rec)
    # Two detector families, and BOTH are needed even for a single round.
    # Round 1's outcome is deterministic from the reset and stands alone; the
    # final data readout is then compared against the LAST round. Folding the
    # two into one detector cancels them (a data fault before the round flips
    # the ancilla and the readout alike) and every circuit reads d_circ = 1.
    for r in range(rounds):
        for cid in range(len(checks)):
            targets = [rec(anc_rec[r][cid])]
            if r:
                targets.append(rec(anc_rec[r - 1][cid]))
            c.append("DETECTOR", targets)
    for cid, row in enumerate(checks):
        c.append("DETECTOR", [rec(anc_rec[rounds - 1][cid])]
                 + [rec(d_rec[q]) for q in np.nonzero(row)[0]])
    for i, lg in enumerate(logical_basis(code, basis)):
        c.append("OBSERVABLE_INCLUDE", [rec(d_rec[q]) for q in np.nonzero(lg)[0]], i)
    return c


def deinterleave(code: CSSCode, times, graph: TannerGraph | None = None):
    """The non-interleaved counterpart of Strikis, Browne and Beverland
    (arXiv:2603.05481): all X-checks, then all Z-checks, **preserving the
    relative CNOT order within each check**.

    Their Theorem 1: its circuit distance ``d'_circ`` satisfies
    ``d'_circ <= d_circ``. Since the per-check order is what fixes the residual
    (hook) errors, the counterpart has exactly the same residual set as the
    original --- only the X/Z cross-propagation is removed. Layers are packed
    greedily rank by rank under data-qubit exclusivity; the result is proper
    automatically (every X-CNOT precedes every Z-CNOT).
    """
    graph = graph if graph is not None else TannerGraph.build(code)
    edges = graph.edges
    by_check: dict = {}
    for e, (ck, _q, _ty) in enumerate(edges):
        by_check.setdefault(ck, []).append(e)
    for ck in by_check:
        by_check[ck].sort(key=lambda e: times[e])
    out = np.full(len(edges), -1, dtype=int)
    offset = 0
    for ty in ("X", "Z"):
        chks = [ck for ck in by_check if ck[0] == ty]
        if not chks:
            continue
        busy: set = set()
        prev = {ck: offset - 1 for ck in chks}
        for r in range(max(len(by_check[ck]) for ck in chks)):
            for ck in chks:
                lst = by_check[ck]
                if r >= len(lst):
                    continue
                e = lst[r]
                q = edges[e][1]
                L = max(prev[ck] + 1, offset)
                while (q, L) in busy:
                    L += 1
                out[e] = L
                busy.add((q, L))
                prev[ck] = L
        offset = int(out[[e for ck in chks for e in by_check[ck]]].max()) + 1
    assert (out >= 0).all(), "an edge was left unscheduled"
    return out


def dem_matrices(dem):
    """``(D, O)`` uint8: detectors x mechanisms, observables x mechanisms.

    stim merges mechanisms with identical signatures. Harmless for a minimum
    weight question: a duplicate column can never shorten a solution.
    """
    dem = dem.flattened()
    cols_d, cols_o = [], []
    for inst in dem:
        if inst.type != "error":
            continue
        ds, os_ = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                ds.append(t.val)
            elif t.is_logical_observable_id():
                os_.append(t.val)
        cols_d.append(ds)
        cols_o.append(os_)
    E = len(cols_d)
    D = np.zeros((dem.num_detectors, E), dtype=np.uint8)
    O = np.zeros((dem.num_observables, E), dtype=np.uint8)
    for e, ds in enumerate(cols_d):
        D[ds, e] = 1
    for e, os_ in enumerate(cols_o):
        O[os_, e] = 1
    return D, O


def drop_dead(D, O):
    """Remove mechanisms that flip nothing; they can only pad a solution."""
    live = D.any(0) | O.any(0)
    return D[:, live], O[:, live]


def dem(code, times, graph, basis: str, p: float = 1e-3, idle: bool = True,
        readout_noise: bool = False, rounds: int = 1, meas_idle: bool = False,
        ideal_prep: bool = False):
    """``(circuit, D, O)`` for one schedule in one basis --- the common prefix
    of every bound computation."""
    circ = memory_circuit(code, times, graph, basis=basis, p=p, idle=idle,
                          rounds=rounds, readout_noise=readout_noise,
                          meas_idle=meas_idle, ideal_prep=ideal_prep)
    D, O = dem_matrices(circ.detector_error_model(decompose_errors=False))
    D, O = drop_dead(D, O)
    return circ, D, O
