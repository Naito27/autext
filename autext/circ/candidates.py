"""Where candidate schedules come from.

Every candidate is a valid, verified schedule at the target depth; what differs
between them is the CNOT order within checks and the interleaving, which is
what the circuit distance depends on.

- :func:`bb_schedules`  every Eq. (S24) sandwich of a BB code: ``2 d_A! d_B!``
  of them, all at depth ``d_A + 2 ceil(d_B/2)``.
- :func:`qt_schedules`  depth-``Delta`` QT schedules from the Box 1 greedy
  (seeded) and from CP-SAT (also seeded --- unseeded, CP-SAT is deterministic
  and each instance yields a single schedule).
- :func:`qt_construct`  the objective of Strikis, Browne and Beverland
  (arXiv:2603.05481) imposed as a CONSTRAINT: forbid every suffix (= residual)
  whose distance is below ``tau`` and raise ``tau`` until no depth-``Delta``
  schedule remains. Reaches the optimum of the single-residual proxy.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

__all__ = ["bb_schedules", "qt_load", "qt_schedules", "qt_construct",
           "qt_cut_planes"]


def bb_schedules(code, l, m, a_terms, b_terms, full: bool = False):
    """Yield ``(label, Schedule)`` over the monomial-coloured sandwich freedom.

    ``full=False`` is Eq. (S24) as written: ``a`` and ``a*`` share layers and
    ``perm_b`` fixes both split bands, giving ``2 d_A! d_B!`` schedules.
    ``full=True`` is everything the band argument allows with a monomial
    colouring: ``a*`` permuted independently of ``a``, either ``floor`` or
    ``ceil`` split of the ``b`` colours, and every placement of the ``b`` and
    ``b*`` classes into the ``ceil(d_B/2)`` layers of their band. Labels are
    ``(merge, perm_a, perm_b)`` or
    ``(merge, perm_a, perm_a2, eb, lb, eb2, lb2)`` with the slot lists of
    ``bb_sandwich_schedule``'s ``b_layout``.
    """
    from ..lp_sandwich import bb_sandwich_schedule

    for merge in ("A", "B"):
        da, db = ((len(a_terms), len(b_terms)) if merge == "A"
                  else (len(b_terms), len(a_terms)))
        for pa in itertools.permutations(range(da)):
            if not full:
                for pb in itertools.permutations(range(db)):
                    yield ((merge, pa, pb),
                           bb_sandwich_schedule(code, l, m, a_terms, b_terms,
                                                merge=merge, perm_a=pa, perm_b=pb))
                continue
            w = (db + 1) // 2                       # width of each split band

            def placements(colours):                # colours -> w slots, injectively
                for pos in itertools.permutations(range(w), len(colours)):
                    out = [None] * w
                    for c, k in zip(colours, pos):
                        out[k] = c
                    yield tuple(out)

            for pa2 in itertools.permutations(range(da)):
                for ne in sorted({w, db - w}):
                    for eset in itertools.combinations(range(db), ne):
                        lset = tuple(c for c in range(db) if c not in eset)
                        for eb in placements(eset):
                            for lb in placements(lset):
                                for eb2 in placements(lset):
                                    for lb2 in placements(eset):
                                        lay = (eb, lb, eb2, lb2)
                                        yield ((merge, pa, pa2) + lay,
                                               bb_sandwich_schedule(
                                                   code, l, m, a_terms, b_terms,
                                                   merge=merge, perm_a=pa,
                                                   perm_a2=pa2, b_layout=lay))


def qt_load(root, spec):
    """``(fam, tag, nA, nB, n, k, d_pub, HX, HZ)`` for ``'family:tag'`` under ``root``."""
    from ..qt.tables import FAMILIES, load_mtx, parse_tag

    fam, tag = spec.split(":")
    _a, _b, nA, nB = FAMILIES[fam]
    _g, n, k, d_pub = parse_tag(tag, nA, nB)
    HX = load_mtx(f"{root}/{fam}/HX_{tag}.mtx")
    HZ = load_mtx(f"{root}/{fam}/HZ_{tag}.mtx")
    return fam, tag, nA, nB, n, k, d_pub, HX, HZ


def qt_schedules(HX, HZ, nA, nB, seeds: int = 24, limit: int = 200,
                 cpsat_tl: float = 20.0, cpsat_seeds: int = 6, workers: int = 4):
    """``(code, graph, Delta, [(label, Schedule), ...])`` --- distinct verified
    depth-``Delta`` schedules from the greedy, then from seeded CP-SAT."""
    from ..qt import simple
    from ..qt.exact import solve as cpsat_solve
    from ..qt.tables import _accept, qt_instances

    code, graph, D, insts = qt_instances(HX, HZ, nA, nB)
    seen, out = set(), []

    def keep(inst, label, times):
        sch, _rep = _accept(code, graph, inst, times, D)
        if sch is None:
            return
        key = sch.times.tobytes()
        if key not in seen:
            seen.add(key)
            out.append((label, sch))

    for ii, inst in enumerate(insts):
        for enl in (False, True):
            for s in range(seeds):
                t = simple.run(inst, D, seed=s, enlarge=enl)
                if t is not None:
                    keep(inst, (ii, "greedy", enl, s), t)
                if len(out) >= limit:
                    return code, graph, D, out
    for rep_seed in range(cpsat_seeds):
        for ii, inst in enumerate(insts):
            st, t = cpsat_solve(inst, D, tl=cpsat_tl, workers=workers, seed=rep_seed)
            if st == "FEASIBLE" and t is not None:
                keep(inst, (ii, "cpsat", rep_seed), t)
            if len(out) >= limit:
                return code, graph, D, out
    return code, graph, D, out


def qt_construct(HX, HZ, nA, nB, max_weight: int = 7, insts_cap: int = 24,
                 tl: float = 25.0, attempts: int = 3, workers: int = 8,
                 verbose: bool = True):
    """Raise the single-residual floor as far as depth ``Delta`` allows.

    Enumerates every suffix of every check (both families, since ``d_circ`` is
    the min over bases), scores each by :func:`.residual.residual_distance`,
    then for ``tau`` = floor+1, floor+2, ... forbids all suffixes with distance
    ``< tau`` via ``qt.exact.solve(forbid_tails=...)``. Returns
    ``(tau_best, Schedule, spectrum)`` or ``(None, None, spectrum)``.
    """
    from ..qt.exact import solve as cpsat_solve
    from ..qt.tables import _accept, qt_instances
    from .build import sector
    from .residual import residual_distance

    code, graph, D, insts = qt_instances(HX, HZ, nA, nB)
    byck, eqb = {}, {}
    for e, (ck, q, ty) in enumerate(graph.edges):
        byck.setdefault((ty, ck[1]), []).append(e)
        eqb[e] = q
    t0 = time.time()
    tail_dist, cache = {}, {}
    for (ty, c), es in byck.items():
        basis = "Z" if ty == "X" else "X"        # X-check hooks hurt Z-basis memory
        sec = sector(code, basis)
        assert sec.hook_checks == ty
        if len(es) > max_weight:
            continue
        d = {}
        for r in range(1, len(es)):
            for S in itertools.combinations(es, r):
                qs = frozenset(eqb[e] for e in S)
                if qs not in cache:
                    cache[qs] = residual_distance(sec.H, sec.L, qs, attempts=attempts)
                d[frozenset(S)] = cache[qs]
        tail_dist[(ty, c)] = d
    allv = [v for d in tail_dist.values() for v in d.values()]
    spectrum = {x: allv.count(x) for x in sorted(set(allv))}
    if verbose:
        print(f"    {sum(len(d) for d in tail_dist.values())} suffixes, "
              f"{len(cache)} distinct residuals, spectrum {spectrum} "
              f"({time.time() - t0:.0f}s)", flush=True)
    best = None
    for tau in range(min(allv) + 1, max(allv) + 2):
        fb = {}
        for (_ty, c), d in tail_dist.items():
            bad = [S for S, v in d.items() if v < tau]
            if bad:
                fb.setdefault(c, []).extend(bad)
        t = time.time()
        hit = None
        for ii, inst in enumerate(insts[:insts_cap]):
            st, tt = cpsat_solve(inst, D, tl=tl, workers=workers, seed=0, forbid_tails=fb)
            if st == "FEASIBLE" and tt is not None:
                sch, _rep = _accept(code, graph, inst, tt, D)
                if sch is not None:
                    hit = (ii, sch)
                    break
        if verbose:
            print(f"    tau={tau}: {'feasible (inst %d)' % hit[0] if hit else 'none'}"
                  f" ({time.time() - t:.0f}s)", flush=True)
        if hit is None:
            break
        best = (tau, hit[1])
    if best is None:
        return None, None, spectrum
    return best[0], best[1], spectrum


def qt_cut_planes(HX, HZ, nA, nB, rounds: int = 12, tl: float = 25.0,
                  ext_time: float = 300.0, insts_cap: int = 10 ** 6, workers: int = 8,
                  seed: int = 0, verbose: bool = True):
    """Raise ``d_ext`` itself by learning no-goods from its witnesses.

    The single-residual floor is the wrong objective, because ``d_ext`` is
    realised by COMBINATIONS of residuals. So target ``d_ext`` directly,
    cutting-plane style:

        1  schedule under the current no-goods
        2  in each basis, solve the extended code exactly; its witness names
           the residual columns that together realise d_ext
        3  forbid that combination: at least one of those residuals must not
           be a suffix in the next schedule (a disjunction, `forbid_combos`)
        4  repeat; keep the schedule with the largest min-over-bases d_ext

    A witness that uses no residual columns is a pure data-error logical ---
    the code distance itself --- and cannot be forbidden; that basis is done.
    Each round costs one scheduler solve plus one exact ``d_ext`` per basis,
    and the value it raises is the CERTIFIED floor on ``d_circ``.

    Returns ``(best_d_ext, Schedule, history)`` with ``history`` a list of
    ``(round, d_ext_Z, d_ext_X)``.
    """
    from ..qt.exact import solve as cpsat_solve
    from ..qt.tables import _accept, qt_instances
    from .bounds import ext_distance
    from .build import sector
    from .residual import extended_matrices, tails

    code, graph, D, insts = qt_instances(HX, HZ, nA, nB)
    eidx = {}
    for e, (ck, q, ty) in enumerate(graph.edges):
        eidx[(ty, ck[1], q)] = e
    combos, history, best = [], [], (None, None)
    for rnd in range(rounds):
        t0 = time.time()
        sch = None
        for ii, inst in enumerate(insts[:insts_cap]):
            st, tt = cpsat_solve(inst, D, tl=tl, workers=workers, seed=seed + rnd,
                                 forbid_combos=combos)
            if st == "FEASIBLE" and tt is not None:
                cand, _rep = _accept(code, graph, inst, tt, D)
                if cand is not None:
                    sch = cand
                    break
        if sch is None:
            if verbose:
                print(f"    round {rnd}: no depth-{D} schedule under "
                      f"{len(combos)} no-goods -- stopping", flush=True)
            break
        vals = {}
        for basis in ("Z", "X"):
            sec = sector(code, basis)
            tl_by_check = tails(code, sch.times, graph, basis)
            cols, seen = [], {}
            for c, lst in tl_by_check.items():
                for sup in lst:
                    if sup in seen:
                        continue
                    seen[sup] = (c, sup)
                    cols.append(sup)
            R = np.zeros((code.n, len(cols)), dtype=np.uint8)
            for j, sup in enumerate(cols):
                R[list(sup), j] = 1
            He, Le = extended_matrices(sec.H, sec.L, R)
            b = ext_distance(He, Le, time_limit=ext_time, workers=workers, basis=basis)
            if b is None or b.kind != "exact":
                vals[basis] = None
                continue
            vals[basis] = b.value
            used = [j for j in np.nonzero(b.witness[code.n:])[0]]
            if not used:
                continue                       # pure data logical: cannot forbid
            combo = []
            for j in used:
                c, sup = seen[cols[j]]
                S = frozenset(eidx[(sec.hook_checks, c, q)] for q in sup)
                combo.append((c, S))
            combos.append(combo)
        history.append((rnd, vals.get("Z"), vals.get("X")))
        good = [v for v in vals.values() if v is not None]
        floor = min(good) if len(good) == 2 else None
        if verbose:
            print(f"    round {rnd}: d_ext Z={vals.get('Z')} X={vals.get('X')} "
                  f"-> floor {floor}; {len(combos)} no-goods "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if floor is not None and (best[0] is None or floor > best[0]):
            best = (floor, sch)
    return best[0], best[1], history
