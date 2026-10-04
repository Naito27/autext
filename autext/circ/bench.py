"""Reproducible circuit-distance runs. Each mode writes
``results/run/circ_<mode>.log``.

    python -m autext.circ.bench validate
        stim's rotated surface code, d = 3 and 5: stim search, CP-SAT and
        BP-OSD must agree. The pipeline's oracle.

    python -m autext.circ.bench certify [--ref36 DIR] [--time SECS] [--codes ...]
        The certified table: [[72,12,6]] (Eq. (S24) sandwich) and the
        60_2_8, 72_2_9, 80_2_10 codes of Leverrier, Rozendaal and Zemor
        (arXiv:2512.20532) (Box 1 greedy / CP-SAT), three circuits each
        --- ours, the non-interleaved counterpart, the staggered baseline ---
        both bases, CP-SAT under a BP-OSD cap.

    python -m autext.circ.bench optimise lp TAG | qt FAM:TAG [--ref36 DIR]
        The staged pipeline of :mod:`.search` over the schedule space.

    python -m autext.circ.bench construct qt FAM:TAG [--ref36 DIR]
        Raise the single-residual floor by constraint, then evaluate.

    python -m autext.circ.bench local FAM:TAG [--iters N] [--ref36 DIR]
        Witness-driven local search (PropHunt-style) on the interleaved circuit
        from the best screened start; then certify the result.

    python -m autext.circ.bench planes FAM:TAG [--rounds N] [--ref36 DIR]
        Cutting planes on d_ext: forbid the residual combinations that realise
        each witness, re-schedule, repeat. Raises the CERTIFIED floor.

Leverrier et al.'s matrices are resolved from ``--ref36``, then
``$AUTEXT_REF36``, then ``results/ref36`` --- they are CC-BY-4.0 and not
redistributed here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .bounds import bracket, exact_cpsat, min_over_bases, upper_bposd, upper_stim
from .build import deinterleave, dem
from .search import _ext_estimate as _ext_estimate_raw


def _ext_estimate(code, times, graph, basis, attempts=12):
    return _ext_estimate_raw(code, times, graph, basis, attempts)

LP_CERT = ["[[72,12,6]]"]
QT_CERT = ["633x212:60_2_8", "633x212:72_2_9", "844x212:80_2_10"]


class Log:
    def __init__(self, mode):
        os.makedirs("results/run", exist_ok=True)
        self.f = open(f"results/run/circ_{mode}.log", "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def ref36_root(arg):
    for cand in (arg, os.environ.get("AUTEXT_REF36"), "results/ref36"):
        if cand and os.path.isdir(cand):
            return cand
    raise SystemExit("QT code matrices not found: pass --ref36 DIR or set AUTEXT_REF36")


# ------------------------------------------------------------------ validate
def validate(log):
    import stim

    log(f"{'d':>2} {'stim':>5} {'cpsat':>6} {'bposd':>6}  agree")
    ok = True
    for d in (3, 5):
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=1,
            after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3,
            after_reset_flip_probability=1e-3, before_round_data_depolarization=1e-3)
        from .build import dem_matrices, drop_dead
        D, O = drop_dead(*dem_matrices(circ.detector_error_model(decompose_errors=False)))
        s = len(circ.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=4,
            dont_explore_edges_increasing_symptom_degree=False))
        c = exact_cpsat(D, O, time_limit=120, ub=d)
        b = upper_bposd(D, O, attempts=10)
        agree = (c is not None and c.kind == "exact" and c.value == s == d
                 and b is not None and b.value == d)
        ok &= agree
        log(f"{d:>2} {s:>5} {str(c.value if c else None):>6} "
            f"{str(b.value if b else None):>6}  {'yes' if agree else 'NO'}")
    log("VALIDATION " + ("OK" if ok else "FAILED"))
    return ok


# ------------------------------------------------------------------- certify
def certify_one(log, label, code, graph, ours, d_code, time_limit, workers=8,
                ext_time=None, hint=True):
    """Three circuits, both bases, CHEAPEST FIRST; the DEM solver runs last and
    only if the cheap bracket did not already close.

        1  upper_stim (lim 4)          hi
        2  upper_bposd on the DEM      hi, and a witness
        3  lower_ext (exact d_ext)     lo (Thm 1) -- ONLY if its cheap BP-OSD
                                       estimate is >= hi, i.e. it could close
                                       the bracket
        4  exact_cpsat on the DEM      only if lo != hi

    Returns ``{circuit: Bracket}``.
    """
    from ..baselines import staggered_schedule
    from .bounds import Bound, lower_ext

    ext_time = ext_time if ext_time is not None else time_limit
    variants = {"ours": ours,
                "non-interleaved": deinterleave(code, ours, graph),
                "staggered": staggered_schedule(code).times}
    out = {}
    for name, times in variants.items():
        per = {}
        for basis in ("Z", "X"):
            t = time.time()
            trivial = (Bound(int(d_code), "upper", "d_circ", "d_code", 0.0, basis)
                       if d_code is not None else None)
            st = upper_stim(code, times, graph, basis, lim=4)
            _c, D, O = dem(code, times, graph, basis)
            ub = upper_bposd(D, O, attempts=30, basis=basis)
            hi = bracket(st, ub, trivial, basis=basis).hi
            # Gate the exact d_ext on its own cheap estimate. BP-OSD's value is
            # an upper bound on d_ext, so if it is already BELOW the circuit's
            # upper bound the exact d_ext cannot meet it and cannot close the
            # bracket, so computing it would be pure loss. Only rows where it
            # might close the bracket pay for it.
            est = _ext_estimate(code, times, graph, basis)
            lo = None
            if hi is None or est is None or est.value >= hi:
                lo = lower_ext(code, times, graph, basis, time_limit=ext_time,
                               workers=workers)
            br = bracket(st, ub, trivial, lo, basis=basis)
            how = "cheap"
            if br.exact is None:
                cap = br.hi
                ex = exact_cpsat(D, O, time_limit=time_limit, workers=workers,
                                 ub=cap, basis=basis,
                                 hint=(ub.witness if (hint and ub is not None) else None))
                br = bracket(st, ub, trivial, lo, ex, basis=basis)
                how = "cpsat"
            per[basis] = br
            log(f"{label:>14} {name:>16} {int(np.max(times)) + 1:>5} {basis:>5} "
                f"{D.shape[1]:>5} {str(st.value if st else None):>5} "
                f"{str(ub.value if ub else None):>5} {str(lo.value if lo else None):>5} "
                f"{str(br):>22} {how:>6} {time.time() - t:>7.1f}")
        out[name] = min_over_bases(per["Z"], per["X"])
    return out


def certify(log, ref36, time_limit, codes):
    from ..distance import probabilistic_distance

    log(f"{'code':>14} {'circuit':>16} {'depth':>5} {'basis':>5} {'mech':>5} "
        f"{'stim':>5} {'bposd':>5} {'d_ext':>5} {'bracket':>22} {'how':>6} {'secs':>7}")
    table = {}
    for tag in codes:
        if tag.startswith("[["):
            from ..bb import CATALOGUE, bb_code
            from ..lp_sandwich import bb_sandwich_schedule
            l, m, a, b = CATALOGUE[tag]
            code = bb_code(l, m, a, b, name=tag)
            d_code = probabilistic_distance(code, trials=300, seed=0)[0]
            sch = bb_sandwich_schedule(code, l, m, a, b)
            graph, ours = sch.graph, sch.times
        else:
            from ..qt.tables import schedule_published
            from .candidates import qt_load
            _f, t2, nA, nB, n, k, d_code, HX, HZ = qt_load(ref36_root(ref36), tag)
            code, _D, sch, rep, how = schedule_published(HX, HZ, nA, nB, n // (nA * nB),
                                                         seeds=64, name=t2)
            if sch is None:
                log(f"{tag}: no verified schedule ({how})")
                continue
            graph, ours, tag = sch.graph, sch.times, t2
        rep = sch.verify()
        assert rep["proper"] and rep["contam"] and rep["sim"], f"{tag} schedule invalid"
        table[tag] = (code.n, code.k, d_code, sch.depth,
                      certify_one(log, tag, code, graph, ours, d_code, time_limit))
    log("")
    log(f"{'code':>14} {'n':>4} {'k':>3} {'d':>4} {'depth':>5} {'ours':>16} "
        f"{'non-interleaved':>16} {'staggered':>16}")
    for tag, (n, k, d, depth, res) in table.items():
        log(f"{tag:>14} {n:>4} {k:>3} {str(d):>4} {depth:>5} {str(res['ours']):>16} "
            f"{str(res['non-interleaved']):>16} {str(res['staggered']):>16}")
    return table


# ------------------------------------------------------------------ optimise
def planes(log, spec, ref36, rounds):
    from .candidates import qt_cut_planes, qt_load

    _f, tag, nA, nB, n, k, d_pub, HX, HZ = qt_load(ref36_root(ref36), spec)
    log(f"{tag}: n={n} k={k} d<={d_pub}; cutting planes on d_ext, {rounds} rounds")
    best, sch, hist = qt_cut_planes(HX, HZ, nA, nB, rounds=rounds)
    for rnd, z, x in hist:
        log(f"  round {rnd}: d_ext Z={z} X={x}")
    log(f"best certified floor d_ext = {best}")
    if sch is not None:
        from ..core import TannerGraph
        graph = TannerGraph.build(sch.code)
        for b in ("Z", "X"):
            log(f"  {b}: {upper_stim(sch.code, sch.times, graph, b, lim=4)}")
        np.save("results/run/circ_planes_best.npy", sch.times)


def local(log, spec, ref36, iters, seed, start_seeds, time_limit):
    """Witness-driven local search from the best screened schedule, then certify."""
    from ..qt.tables import _accept, qt_instances
    from ..schedule import Schedule
    from .bounds import Bound, lower_ext
    from .candidates import qt_load, qt_schedules
    from .local import local_search, score
    from .search import screen

    _f, tag, nA, nB, n, k, d_pub, HX, HZ = qt_load(ref36_root(ref36), spec)
    code, graph, D, insts = qt_instances(HX, HZ, nA, nB)
    _c, _g, _D, cs = qt_schedules(HX, HZ, nA, nB, seeds=24, limit=60,
                                  cpsat_tl=15.0, cpsat_seeds=start_seeds)
    log(f"{tag}: n={code.n} k={code.k} d<={d_pub} Delta={D}; "
        f"{len(cs)} start candidates")
    sc = screen(code, graph, [(lab, s_.times) for lab, s_ in cs], verbose=False)
    order = sorted(sc, key=lambda l: tuple(v if v is not None else -1 for v in sc[l]),
                   reverse=True)
    lab = order[0]
    sch0 = dict(cs)[lab]
    inst = insts[lab[0]]
    log(f"start {lab}: screen {sc[lab]}")

    t0 = time.time()
    best1, best_s, hist = local_search(inst, code, graph, sch0.times + 1, iters=iters,
                                       seed=seed, verbose=True)
    log(f"local search: witness weight {sc[lab][1]} -> {best_s} in {iters} iters "
        f"({time.time() - t0:.0f}s); improvements at {hist}")
    sch, rep = _accept(code, graph, inst, best1, D)
    assert sch is not None and rep["proper"] and rep["contam"] and rep["sim"]
    np.save(f"results/run/circ_local_best_{tag}.npy", sch.times)   # per code: runs overwrite otherwise

    per = {}
    for b in ("Z", "X"):
        t = time.time()
        trivial = Bound(int(d_pub), "upper", "d_circ", "d_code", 0.0, b)
        st = upper_stim(code, sch.times, graph, b, lim=4)
        st5 = upper_stim(code, sch.times, graph, b, lim=5)
        _cc, Dm, Om = dem(code, sch.times, graph, b)
        ub = upper_bposd(Dm, Om, attempts=30, basis=b)
        lo = lower_ext(code, sch.times, graph, b, time_limit=time_limit)
        br = bracket(st, st5, ub, trivial, lo, basis=b)
        how = "cheap"
        if br.exact is None:
            ex = exact_cpsat(Dm, Om, time_limit=time_limit, ub=br.hi, basis=b,
                             hint=(ub.witness if ub is not None else None))
            br = bracket(st, st5, ub, trivial, lo, ex, basis=b)
            how = "cpsat"
        per[b] = br
        log(f"  {b}: stim4={st.value if st else None} stim5={st5.value if st5 else None} "
            f"bposd={ub.value if ub else None} d_ext={lo.value if lo else None} "
            f"-> {br} [{how}, {time.time() - t:.0f}s]")
    tot = min_over_bases(per["Z"], per["X"])
    log(f"RESULT {tag}: {tot}")
    return tot


def optimise(log, kind, spec, ref36, cpsat_seeds=8, limit=150):
    from .candidates import bb_schedules, qt_load, qt_schedules
    from .search import pipeline

    if kind == "lp":
        from ..bb import CATALOGUE, bb_code
        from ..lp_sandwich import bb_sandwich_schedule
        l, m, a, b = CATALOGUE[spec]
        code = bb_code(l, m, a, b, name=spec)
        graph = bb_sandwich_schedule(code, l, m, a, b).graph
        cands = [(lab, s.times) for lab, s in bb_schedules(code, l, m, a, b)]
        log(f"{spec}: {len(cands)} sandwich schedules")
    else:
        _f, tag, nA, nB, n, k, d_pub, HX, HZ = qt_load(ref36_root(ref36), spec)
        code, graph, D, cs = qt_schedules(HX, HZ, nA, nB, seeds=24, limit=limit,
                                          cpsat_tl=15.0, cpsat_seeds=cpsat_seeds)
        cands = [(lab, s.times) for lab, s in cs]
        log(f"{tag}: n={code.n} k={code.k} d<={d_pub} Delta={D}; "
            f"{len(cands)} distinct depth-{D} schedules")
    res = pipeline(code, graph, cands, verbose=True)
    log(f"winner {res['winner']}: {res['total']}")
    for b in ("Z", "X"):
        log(f"  {b}: {res['per_basis'][b]}")
    return res


def construct(log, spec, ref36):
    from .candidates import qt_construct, qt_load

    _f, tag, nA, nB, n, k, d_pub, HX, HZ = qt_load(ref36_root(ref36), spec)
    tau, sch, spectrum = qt_construct(HX, HZ, nA, nB)
    log(f"{tag}: residual spectrum {spectrum}; best floor tau={tau}")
    if sch is None:
        return
    from ..core import TannerGraph
    graph = TannerGraph.build(sch.code)
    for b in ("Z", "X"):
        u = upper_stim(sch.code, sch.times, graph, b, lim=4)
        log(f"  {b}: {u}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m autext.circ.bench")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("validate")
    c = sub.add_parser("certify")
    c.add_argument("--ref36")
    c.add_argument("--time", type=float, default=1200.0)
    c.add_argument("--codes", nargs="*", default=LP_CERT + QT_CERT)
    o = sub.add_parser("optimise")
    o.add_argument("kind", choices=["lp", "qt"])
    o.add_argument("spec")
    o.add_argument("--ref36")
    o.add_argument("--cpsat-seeds", type=int, default=8)
    o.add_argument("--limit", type=int, default=150)
    lo = sub.add_parser("local")
    lo.add_argument("spec")
    lo.add_argument("--ref36")
    lo.add_argument("--iters", type=int, default=300)
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--start-seeds", type=int, default=8)
    lo.add_argument("--time", type=float, default=400.0)
    pl = sub.add_parser("planes")
    pl.add_argument("spec")
    pl.add_argument("--ref36")
    pl.add_argument("--rounds", type=int, default=12)
    k = sub.add_parser("construct")
    k.add_argument("spec")
    k.add_argument("--ref36")
    a = ap.parse_args(argv)
    log = Log(a.mode)
    if a.mode == "validate":
        return 0 if validate(log) else 1
    if a.mode == "certify":
        certify(log, a.ref36, a.time, a.codes)
    elif a.mode == "optimise":
        optimise(log, a.kind, a.spec, a.ref36, a.cpsat_seeds, a.limit)
    elif a.mode == "planes":
        planes(log, a.spec, a.ref36, a.rounds)
    elif a.mode == "local":
        local(log, a.spec, a.ref36, a.iters, a.seed, a.start_seeds, a.time)
    elif a.mode == "construct":
        construct(log, a.spec, a.ref36)
    return 0


if __name__ == "__main__":
    sys.exit(main())
