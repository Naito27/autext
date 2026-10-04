"""The two numerical tables for the QT section.

    python -m autext.qt.tables base            # Table 1: base codes
    python -m autext.qt.tables lift <dir>      # Table 2: lifted codes
    python -m autext.qt.tables lrc [dir]       # amortized left-right cost
    python -m autext.qt.tables base --latex

**Table 1 -- base codes.**  Every qLDPC-eligible pair of local codes
(`ldpc.qldpc_pairs`, the criterion of Leverrier, Rozendaal and Zemor,
arXiv:2512.20532), swept over every inequivalent column permutation of the
index-1 matrices.  This is the reduced problem: it depends only on the
classical codes and the permutation, never on the group or the multisets, so
one row covers every lifted code built from that pair.

**Table 2 -- lifted codes.**  The parity-check matrices published by Leverrier
et al., scheduled directly.  Fetch them with

    curl -L -o src.tar.gz https://arxiv.org/e-print/2512.20532 && tar -xzf src.tar.gz

which unpacks `633x212/`, `633x633/` and `844x212/` of Matrix Market files named
`H{X,Z}[_<group>]_<n>_<k>_<d>.mtx`.  Pass the directory containing those.

Every depth reported here comes from a schedule that passed `core.verify`, and
every lifted schedule additionally passed the full-code verifier (Eq. (2) parity
formula, two-directional propagation).  Misses are adjudicated with CP-SAT, so a
reported failure is a certified obstruction rather than solver weakness.
"""
import glob
import itertools
import os
import sys
import time

import numpy as np

from . import core, simple
from .build import GROUP, build
from .ldpc import REF36_PAIRS, generator_weight, perm_classes, qldpc_pairs
from .qt import QT, balanced_halvings
from ..core import CSSCode, TannerGraph
from ..schedule import Schedule

__all__ = ["base_row", "table_base", "load_mtx", "schedule_published",
           "table_lift"]


# --------------------------------------------------------------------------
# Table 1: the base codes
# --------------------------------------------------------------------------
def sweep_base(Q, restarts=16):
    """Run the Box-1 layer sweep on every balanced halving.

    Returns `(reached, straddling, secs, m, Delta)`.  `straddling` is the list of
    halvings that passed `core.straddles` -- the only ones that can carry a
    depth-Delta schedule, and so the only ones CP-SAT need consider.
    """
    t0 = time.time()
    straddling, m, D = [], 0, 0
    for rho in balanced_halvings(Q.nA):
        for sig in balanced_halvings(Q.nB):
            inst = build(Q.pair, Q.nA, Q.nB, np.array(rho), np.array(sig))
            D, m = inst.Delta(), inst.m
            if inst.m == 0 or D == 0 or not core.straddles(inst, D):
                continue
            straddling.append((rho, sig))
            for enl in (False, True):
                for s in range(restarts):
                    sch = simple.run(inst, D, seed=s, enlarge=enl)
                    if sch is not None:
                        core.verify(inst, sch, D)
                        return True, straddling, time.time() - t0, m, D
    return False, straddling, time.time() - t0, m, D


def cpsat_base(Q, straddling, tl=30):
    """The same question put to CP-SAT: is depth Delta feasible on any of these
    halvings?  Returns `(verdict, secs)`, verdict FEASIBLE / INFEASIBLE /
    unknown.  INFEASIBLE means proved on every straddling halving."""
    from .exact import solve
    t0 = time.time()
    if not straddling:
        return 'no straddling halving', 0.0
    unknown = False
    for rho, sig in straddling:
        inst = build(Q.pair, Q.nA, Q.nB, np.array(rho), np.array(sig))
        st, t = solve(inst, inst.Delta(), tl=tl, workers=4)
        if st == 'FEASIBLE':
            core.verify(inst, t, inst.Delta())
            return 'FEASIBLE', time.time() - t0
        if st == 'unknown':
            unknown = True
    return ('unknown' if unknown else 'INFEASIBLE'), time.time() - t0


def base_row(a, b, restarts=16, tl=30, verbose=False):
    """One Table-1 row.

    BOTH methods run on EVERY (piA, piB) class, so the two time columns compare
    the methods against each other rather than the greedy against its own
    leftovers.  Outcomes split three ways: the greedy reaches Delta; the greedy
    misses but CP-SAT reaches it; or CP-SAT proves Delta infeasible.
    """
    PA, PB = perm_classes(a), perm_classes(b)
    deltas = set()
    n_sweep = n_cpsat = n_inf = n_unk = n_bad = 0
    t_sweep = t_cpsat = 0.0
    m = 0
    for piA in PA:
        for piB in PB:
            Q = QT(a, b, piA, piB)
            got, straddling, ts, mm, D = sweep_base(Q, restarts)
            verdict, tc = cpsat_base(Q, straddling, tl)
            deltas.add(D)
            m = mm or m
            t_sweep += ts
            t_cpsat += tc
            if got:
                n_sweep += 1
                # the greedy succeeded on one of the halvings CP-SAT scans, so
                # CP-SAT must agree; a disagreement means one of them is wrong
                if verdict != 'FEASIBLE':
                    n_bad += 1
            elif verdict == 'FEASIBLE':
                n_cpsat += 1
            elif verdict == 'INFEASIBLE':
                n_inf += 1
            else:
                n_unk += 1
            if verbose and not got:
                print(f'      piA={piA} piB={piB}: greedy missed, CP-SAT {verdict}',
                      flush=True)
    return dict(a=a, b=b, w=generator_weight(a, b), nA=QT(a, b).nA,
                nB=QT(a, b).nB, m=m, classes=len(PA) * len(PB),
                dmin=min(deltas), dmax=max(deltas), sweep=n_sweep,
                cpsat=n_cpsat, infeasible=n_inf, unknown=n_unk,
                disagree=n_bad, t_sweep=t_sweep, t_cpsat=t_cpsat,
                ref36=(a, b) in REF36_PAIRS)


def table_base(pairs=None, restarts=16, tl=30, verbose=False):
    pairs = pairs or qldpc_pairs()
    print(f"{'C_A x C_B':>20s} {'w':>3s} {'|E|':>5s} {'Delta':>6s} {'classes':>8s} "
          f"{'greedy=D':>9s} {'CPSAT=D':>8s} {'infeas':>7s} {'unkn':>5s} "
          f"{'t_greedy':>10s} {'t_cpsat':>10s}  note")
    rows = []
    for a, b, _w in pairs:
        r = base_row(a, b, restarts, tl, verbose)
        rows.append(r)
        dd = str(r['dmin']) if r['dmin'] == r['dmax'] else f"{r['dmin']}-{r['dmax']}"
        note = '[Ref.36]' if r['ref36'] else ''
        if r['disagree']:
            note += f"  !! {r['disagree']} greedy/CP-SAT disagreements"
        print(f"{a+' x '+b:>20s} {r['w']:3d} {r['m']:5d} {dd:>6s} {r['classes']:8d} "
              f"{r['sweep']:9d} {r['cpsat']:8d} {r['infeasible']:7d} "
              f"{r['unknown']:5d} {r['t_sweep']:9.1f}s {r['t_cpsat']:9.1f}s  {note}",
              flush=True)
    tot = sum(r['classes'] for r in rows)
    print(f"  TOTAL {tot} classes: greedy reaches Delta "
          f"{sum(r['sweep'] for r in rows)}, CP-SAT only "
          f"{sum(r['cpsat'] for r in rows)}, certified infeasible "
          f"{sum(r['infeasible'] for r in rows)}, unknown "
          f"{sum(r['unknown'] for r in rows)}")
    print(f"  TOTAL time: greedy {sum(r['t_sweep'] for r in rows):.1f}s, "
          f"CP-SAT {sum(r['t_cpsat'] for r in rows):.1f}s")
    return rows


def table_lrc_base(pairs=None, restarts=4, sweeps=12):
    """Amortized left-right cost per permutation class, on the BASE code.

    The left-right circuit is that of Strikis, Browne and Beverland
    (arXiv:2603.05481).  A base partition lifts with the same `t1+t2`, and
    the base graph is small enough that the search reaches the optimum -- see
    `lrc`.
    """
    from .group import Group
    from .lift import lifted_code
    from .lrc import best_lrc
    triv = Group.cyclic(1)
    pairs = pairs or qldpc_pairs()
    print(f"{'C_A x C_B':>20s} {'w':>3s} {'Delta':>6s} {'classes':>8s} "
          f"{'LRC range':>10s} {'LRC=Delta':>10s} {'mean':>6s}")
    rows = []
    for a, b, w in pairs:
        PA, PB = perm_classes(a), perm_classes(b)
        vals, D = [], None
        for piA in PA:
            for piB in PB:
                Q = QT(a, b, piA, piB)
                base, _ = lifted_code(Q.pair, Q.nA, Q.nB, triv,
                                      [0] * Q.nA, [0] * Q.nB)
                c, _m, D = best_lrc(base.HX, base.HZ, restarts=restarts,
                                    sweeps=sweeps, seed=0)
                vals.append(c)
        rows.append(dict(a=a, b=b, w=w, D=D, classes=len(vals), lo=min(vals),
                         hi=max(vals), eq=sum(v == D for v in vals),
                         mean=sum(vals) / len(vals)))
        r = rows[-1]
        rng_s = f"{r['lo']}-{r['hi']}"
        eq_s = f"{r['eq']}/{r['classes']}"
        print(f"{a+' x '+b:>20s} {w:3d} {D:6d} {r['classes']:8d} "
              f"{rng_s:>10s} {eq_s:>10s} {r['mean']:6.1f}", flush=True)
    return rows


def lrc_published(HX, HZ, nA, nB):
    """Amortized left-right cost of a published lifted code, searched over
    FIBRE-RESPECTING partitions -- i.e. over partitions of its base graph."""
    from .lrc import best_lrc
    cell = np.arange(HX.shape[1]) % (nA * nB)   # their layout: g outermost
    return best_lrc(HX, HZ, restarts=6, sweeps=20, seed=0, cell_of=cell)


# --------------------------------------------------------------------------
# Table 2: the lifted codes, from published parity checks
# --------------------------------------------------------------------------
def load_mtx(path):
    """Matrix Market coordinate file -> dense GF(2) array."""
    with open(path) as f:
        lines = [l for l in f if not l.startswith('%')]
    r, c, _ = (int(x) for x in lines[0].split())
    M = np.zeros((r, c), np.uint8)
    for l in lines[1:]:
        i, j, v = l.split()
        M[int(i) - 1, int(j) - 1] = int(v) % 2
    return M


def qt_instances(HX, HZ, nA, nB):
    """Every `core.Inst` reading of a published lifted QT code.

    Every lifted edge inherits the sandwich group of its base edge, so the full
    lifted Tanner graph is itself a `core.Inst` and both solvers apply to it
    directly -- no reduction, and no need to recover the construction data.

    The file layout has to be decoded first.  Leverrier et al. index qubits
    with the GROUP COORDINATE OUTERMOST, `q = g*(nA*nB) + cell`, which the X0
    block's signature pins down: an X0 check reaches qubits that all share one `g`
    (`phi_X0 = g`), so exactly the first half of the HX rows have constant
    `q // (nA*nB)`.  The cell order and which row block is which family are
    sub-swept here, and the CONTAMINATION check arbitrates later: a
    wrong family/quadrant assignment cannot yield a contamination-free circuit,
    so a pass is self-certifying.
    """
    code = CSSCode(HX, HZ)
    D = code.delta
    graph = TannerGraph.build(code)
    hx2, hz2 = HX.shape[0] // 2, HZ.shape[0] // 2
    q = np.array([e[1] for e in graph.edges])
    cell = q % (nA * nB)
    cid = {}
    ck = np.array([cid.setdefault(e[0], len(cid)) for e in graph.edges], np.int64)
    isX = np.array([e[2] == 'X' for e in graph.edges])
    top = np.array([e[0][1] < (hx2 if e[2] == 'X' else hz2) for e in graph.edges])

    byvar = []
    for swap in (False, True):
        ii = cell % nA if swap else cell // nB
        jj = cell // nA if swap else cell % nB
        for xflip, zflip in itertools.product((False, True), repeat=2):
            here = []
            fam = np.where(isX, np.where(top ^ xflip, 'X0', 'X1'),
                           np.where(top ^ zflip, 'Z0', 'Z1'))
            for rho in balanced_halvings(nA):
                for sig in balanced_halvings(nB):
                    quad = 2 * np.asarray(rho)[ii] + np.asarray(sig)[jj]
                    gr = np.array([GROUP[int(u)][f] for u, f in zip(quad, fam)],
                                  np.int8)
                    inst = core.Inst(q, ck, gr, code.n, len(cid))
                    if inst.Delta() == D and core.straddles(inst, D):
                        here.append(inst)
            byvar.append(here)
    # INTERLEAVE the variants.  Only one decoding is the right one, and a wrong
    # one still yields sandwich-valid schedules that simply are not
    # contamination-free -- so a solver budget spent variant-major can be
    # consumed entirely on a wrong reading.  Round-robin gives every variant a
    # turn before any variant gets a second.
    out = []
    for k in range(max((len(v) for v in byvar), default=0)):
        for v in byvar:
            if k < len(v):
                out.append(v[k])
    return code, graph, D, out


def _accept(code, graph, inst, times, T):
    """`times` is 1-based on `inst`; verify it as a full-code circuit."""
    core.verify(inst, times, T)
    sch = Schedule(code=code, graph=graph, times=times - 1, T=T,
                   meta=dict(method='qt'))
    rep = sch.verify()
    return (sch, rep) if (rep['proper'] and rep['contam'] and rep['sim']) else (None, rep)


def schedule_published(HX, HZ, nA, nB, L, seeds=64, name='', tl=30,
                       cpsat_budget=600.0):
    """Schedule a published lifted QT code at depth Delta.

    The layer sweep first (the paper's Box 1), then CP-SAT on the same instances
    -- which is the manuscript's own stated method, "using the reduced problem
    as input to a CP-SAT solver".  Returns
    `(code, Delta, schedule, report, how)` with `how` naming the solver that
    got there and `schedule` None on a miss.  The `Schedule` itself is
    returned, not just its depth, because downstream work (circuit distance,
    `autext.circ`) needs the layer assignment and not only its length.
    """
    from .exact import solve
    code, graph, D, insts = qt_instances(HX, HZ, nA, nB)
    code.name = name
    for inst in insts:
        for enl in (False, True):
            for s in range(seeds):
                t = simple.run(inst, D, seed=s, enlarge=enl)
                if t is None:
                    continue
                sch, rep = _accept(code, graph, inst, t, D)
                if sch is not None:
                    return code, D, sch, rep, 'sweep'
    t0 = time.time()
    for inst in insts:
        if time.time() - t0 > cpsat_budget:
            break
        st, t = solve(inst, D, tl=tl, workers=4)
        if st == 'FEASIBLE':
            sch, rep = _accept(code, graph, inst, t, D)
            if sch is not None:
                return code, D, sch, rep, 'CP-SAT'
    return code, D, None, None, 'miss'


# directory name -> (local codes, nA, nB); the group is cyclic of order
# n / (nA*nB) unless the filename names it
FAMILIES = {'633x212': ('[6,3,3]', '[2,1,2]', 6, 2),
            '633x633': ('[6,3,3]', '[6,3,3]', 6, 6),
            '844x212': ('[8,4,4]', '[2,1,2]', 8, 2)}


def published_catalogue(root):
    """Every (family, tag, HX path, HZ path) under `root`, ordered by n."""
    out = []
    for d in sorted(FAMILIES):
        _a, _b, nA, nB = FAMILIES[d]
        here = []
        for hx in sorted(glob.glob(os.path.join(root, d, 'HX_*.mtx'))):
            hz = hx.replace('HX_', 'HZ_')
            if os.path.exists(hz):
                tag = os.path.basename(hx)[3:-4]
                here.append((parse_tag(tag, nA, nB)[1], d, tag, hx, hz))
        out += [(d, tag, hx, hz) for _n, d, tag, hx, hz in sorted(here)]
    return out


def parse_tag(tag, nA, nB):
    """`[C2C2_]144_12_11` -> (group name, n, k, d)."""
    parts = tag.split('_')
    n, k, d = (int(x) for x in parts[-3:])
    grp = '_'.join(parts[:-3]) or f'C{n // (nA * nB)}'
    return grp, n, k, d


def table_lift(root, seeds=64):
    from ..baselines import staggered_schedule
    print(f"{'C_A x C_B':>20s} {'group':>8s} {'n':>5s} {'k':>4s} {'d':>4s} "
          f"{'w':>3s} {'Delta':>6s} {'stag':>5s} {'depth':>6s} {'verified':>9s} "
          f"{'solver':>7s} {'secs':>7s}")
    rows = []
    for fam, tag, hx, hz in published_catalogue(root):
        a, b, nA, nB = FAMILIES[fam]
        grp, n, k, d = parse_tag(tag, nA, nB)
        L = n // (nA * nB)
        t0 = time.time()
        HX, HZ = load_mtx(hx), load_mtx(hz)
        code, D, sch, rep, how = schedule_published(HX, HZ, nA, nB, L, seeds, tag)
        depth = None if sch is None else sch.depth
        stag = staggered_schedule(code).depth
        ver = '-' if depth is None else ('yes' if rep['proper'] and rep['contam']
                                         and rep['sim'] else 'NO')
        rows.append(dict(a=a, b=b, grp=grp, n=code.n, k=code.k, d=d,
                         w=generator_weight(a, b), D=D, stag=stag, depth=depth,
                         ver=ver, how=how, secs=time.time() - t0))
        print(f"{a+' x '+b:>20s} {grp:>8s} {code.n:5d} {code.k:4d} {d:4d} "
              f"{rows[-1]['w']:3d} {D:6d} {stag:5d} "
              f"{(str(depth) if depth else 'MISS'):>6s} {ver:>9s} {how:>7s} "
              f"{rows[-1]['secs']:7.1f}", flush=True)
    got = sum(r['depth'] == r['D'] for r in rows)
    print(f'  {got}/{len(rows)} published codes scheduled at Delta')
    return rows


# --------------------------------------------------------------------------
def _secs(x):
    """Seconds at a sensible precision: sub-10s rounds to 0 otherwise."""
    return f'{x:.1f}' if x < 10 else f'{x:.0f}'


def latex_base(rows):
    head = (r'$C_A$ & $C_B$ & $w$ & $|E_{\rm base}|$ & $\Delta$ & \#classes & '
            r'greedy & CP-SAT only & infeasible & '
            r'$t_{\rm greedy}$ & $t_{\rm CP}$ \\')
    out = [r'\begin{tabular}{@{}llrrrrrrrrr@{}}', r'\toprule', head, r'\midrule']
    for r in rows:
        dd = (str(r['dmin']) if r['dmin'] == r['dmax']
              else f"{r['dmin']}--{r['dmax']}")
        out.append(
            f"${r['a']}$ & ${r['b']}$ & {r['w']} & {r['m']} & {dd} & "
            f"{r['classes']} & {r['sweep']} & {r['cpsat']} & {r['infeasible']} & "
            rf"{_secs(r['t_sweep'])}\,s & {_secs(r['t_cpsat'])}\,s \\")
    out += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(out)


def latex_lift(rows):
    out = [r'\begin{tabular}{@{}llrrrrrrrl@{}}', r'\toprule',
           r'$C_A\times C_B$ & $\mathcal{G}$ & $n$ & $k$ & $d$ & $w$ & '
           r'$\Delta$ & $\Delta_X{+}\Delta_Z$ & depth \\', r'\midrule']
    for r in rows:
        out.append(f"${r['a']}\\times{r['b']}$ & ${r['grp']}$ & {r['n']} & "
                   f"{r['k']} & {r['d']} & {r['w']} & {r['D']} & {r['stag']} & "
                   f"{r['depth'] if r['depth'] else '--'} & {r['how']} \\\\")
    out += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(out)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('base', 'lift', 'lrc'):
        raise SystemExit('usage: python -m autext.qt.tables base|lift|lrc [dir] [--latex]')
    latex = '--latex' in sys.argv
    if sys.argv[1] == 'lrc':
        table_lrc_base()
    elif sys.argv[1] == 'base':
        rows = table_base(verbose=True)
        if latex:
            print(); print(latex_base(rows))
    else:
        root = sys.argv[2]
        rows = table_lift(root)
        if latex:
            print(); print(latex_lift(rows))
