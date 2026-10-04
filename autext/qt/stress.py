"""The stress battery for the QT scheduling problem -- all regimes.

    python -m autext.qt.stress <battery> [seeds...]

    scale     n_A x n_B growing to 900 squares
    aspect    n_A and n_B very different, both ways round
    rates     random sizes at extreme rates (k = 1, 2, n/2, n-2, n-1)
    expander  Tanner codes of random (c,d)-regular bipartite graphs
    exsweep   the same, small, swept over more seeds
    general   the [A] control: C_0 = C_1 with pi = id, for comparison
    ldpc      the REALISTIC regime: small bounded-weight local codes, scaled by
              the group lift rather than by n -- see below
    mix       all four local codes DIFFERENT (Leverrier, Rozendaal and Zemor,
              arXiv:2512.20532, take C_0, C_1, C'_0, C'_1 independently), swept
              over the catalogue by length

Generality.  Assumption [A] is dropped the way Eq. (S45) allows: each side draws
TWO INDEPENDENT local codes, one per index c, and the families are assembled by
`qt.pairs_from_codes`, so H_0 != H_1 and G_0 != G_1 as codes, not merely as
matrices.  Every builder asserts `qt.commutes`, so what is scheduled is a real
CSS code.  Permuting the four families independently is not a valid way to drop
[A]: commutation ties the relabelling of H_c to that of G_c, so independent
permutations give HX HZ^T != 0, which is a scheduling instance of the right
shape but not a QT code.

Halvings.  Every halving is scheduled and the best gap over Delta is reported.
`core.straddles` is used only to ORDER the candidates: it is a necessary
condition for depth Delta, so a halving failing it cannot give gap 0 and is
worth trying last.  That is a pure speedup; the reported gap is the same either
way.  The `str` column counts the straddling halvings actually examined before
the sweep stopped -- so `str` = 1 next to `gap` = 0 means the first one was
enough, not that only one straddled.

**These batteries are not qLDPC codes, and are not meant to be.**  They scale
the local code length n, but a random code has a DENSE dual (column weight ~n/2),
so with the families using (H, G') and (G, H') the degree grows like n^2.  The
lift preserves degrees exactly, so that is the Delta of the lifted code too:
these are stress tests of the SOLVER on hard bipartite instances, not realistic
codes.

A real quantum Tanner code is the opposite shape: the local codes are short and
bounded-weight (the catalogue in `codes.py` has column and row weights <= 4, so
Delta <= 20) and the size comes from the group lift, which multiplies the qubit
count by |G| and leaves every degree alone.  The base graph -- the thing
scheduled here -- therefore stays SMALL no matter how large the code is, which
is the whole point of inverting the lift.  The `ldpc` battery measures that
axis: fixed small base pair, growing |G|.

Every reported schedule has passed `core.verify`.
"""
import sys
import time
import random

import numpy as np

from . import core, simple
from .build import build
from .expander import expander_code
from .qt import commutes, pairs_from_codes
from .random_codes import random_code, random_code_degenerate

__all__ = ["halving", "side", "code_pairs", "expander_pairs", "attempt"]


def halving(n, rng):
    v = np.zeros(n, int)
    v[rng.permutation(n)[: n // 2]] = 1
    return v


def side(inst):
    """Which side Delta sits on -- the check side is where straddling bites."""
    ds = int(inst.degS.max()); dc = int(inst.degC.max())
    return 'square' if ds > dc else ('check' if dc > ds else 'both')


# --------------------------------------------------------------------------
# builders -- two independent codes per side, one per index c
# --------------------------------------------------------------------------
def code_pairs(nA, kA, nB, kB, seed, degenerate=False, identical=False):
    """Random local codes.  `degenerate` allows zero-weight columns (and skips
    the column scramble); `identical` is the [A] control, C_0 = C_1, pi = id."""
    pyr = random.Random(seed)
    draw = random_code_degenerate if degenerate else random_code
    A0, B0 = draw(pyr, nA, kA), draw(pyr, nB, kB)
    if A0 is None or B0 is None:
        return None
    if identical:
        A1, B1 = A0, B0
    else:
        A1, B1 = draw(pyr, nA, kA), draw(pyr, nB, kB)
        if A1 is None or B1 is None:
            return None
    pairs = pairs_from_codes(A0, A1, B0, B1)
    assert commutes(pairs), 'builder produced a non-CSS pairs dict'
    return pairs


def expander_pairs(nA, cA, dA, nB, cB, dB, seed):
    npr = np.random.default_rng(seed)
    A0, A1 = expander_code(nA, cA, dA, npr), expander_code(nA, cA, dA, npr)
    B0, B1 = expander_code(nB, cB, dB, npr), expander_code(nB, cB, dB, npr)
    if None in (A0, A1, B0, B1):
        return None
    pairs = pairs_from_codes(A0, A1, B0, B1)
    assert commutes(pairs), 'builder produced a non-CSS pairs dict'
    return pairs


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def attempt(pairs, nA, nB, seed, tries=16, seeds=8, span=3, budget=900.0):
    """Best gap over Delta the layer sweep reaches on any of `tries` halvings."""
    npr = np.random.default_rng(seed)
    t0 = time.time()
    hv = [(halving(nA, npr), halving(nB, npr)) for _ in range(tries)]
    state = dict(best=None, D=0, m=0, sd='-', nstr=0, out=None)

    def try_one(inst):
        """Schedule one instance; True to stop the whole sweep.

        Each depth is tried without the maximum-matching enlargement first and
        with it only if that misses -- the cheap pass is what scales (see
        `simple.solve`), and falling back cannot change the depth reached."""
        D0 = inst.Delta()
        for k in range(span):
            if state['best'] is not None and k >= state['best']:
                break
            got = False
            for enl in (False, True):
                for s in range(seeds):
                    if time.time() - t0 > budget:
                        state['out'] = 'timeout'
                        return True
                    sch = simple.run(inst, D0 + k, seed=s, enlarge=enl)
                    if sch is not None:
                        core.verify(inst, sch, D0 + k)
                        if state['best'] is None or k < state['best']:
                            state.update(best=k, D=D0, m=inst.m, sd=side(inst))
                        got = True
                        break
                if got:
                    break
            if state['best'] == 0:
                return True
        return False

    # Two passes, one instance live at a time -- pre-building every candidate in
    # order to sort them would cost `tries` times the memory of one.  Pass 1
    # takes the straddling halvings, which are the only ones that can give
    # gap 0; pass 2 revisits the rest.  No halving is filtered out, so the
    # reported gap is the same as scheduling them all.
    deferred = []
    for rho, sig in hv:
        inst = build(pairs, nA, nB, rho, sig)
        if inst.m == 0 or inst.Delta() == 0:
            return None
        if not state['m']:
            state.update(D=inst.Delta(), m=inst.m, sd=side(inst))
        if not core.straddles(inst, inst.Delta()):
            deferred.append((rho, sig))
            continue
        state['nstr'] += 1
        if try_one(inst):
            break
    else:
        for rho, sig in deferred:
            if try_one(build(pairs, nA, nB, rho, sig)):
                break

    note = state['out'] or ('verified' if state['best'] is not None else 'no fit')
    return dict(D=state['D'], m=state['m'], gap=state['best'],
                nstr=state['nstr'], side=state['sd'],
                secs=time.time() - t0, note=note)


def key(r):
    """Histogram key: the gap, or why there is none."""
    return r['gap'] if r['gap'] is not None else r['note']


# --------------------------------------------------------------------------
# batteries
# --------------------------------------------------------------------------
CODE_CASES = {
    'scale':  [(8, 4, 8, 4), (12, 6, 12, 6), (16, 8, 16, 8), (20, 10, 20, 10),
               (24, 12, 24, 12), (28, 14, 24, 12), (30, 15, 30, 15)],
    'aspect': [(2, 1, 40, 20), (4, 1, 64, 32), (64, 32, 4, 1), (5, 2, 60, 30),
               (4, 2, 100, 50), (100, 50, 4, 2), (8, 4, 80, 40),
               (80, 40, 8, 4), (12, 6, 60, 30)],
}
EXP_CASES = {
    'expander': [(32, 3, 6, 32, 3, 6), (40, 3, 6, 40, 3, 6),
                 (40, 4, 8, 40, 4, 8), (48, 3, 6, 48, 3, 6)],
    'exsweep':  [(8, 3, 4), (12, 3, 6), (16, 3, 6), (16, 4, 8), (20, 3, 4),
                 (20, 3, 5), (24, 3, 6), (24, 4, 8)],
}
BATTERIES = ('scale', 'aspect', 'rates', 'expander', 'exsweep', 'general',
              'ldpc', 'mix')

# the realistic regime: bounded-weight local codes, size from the lift
LDPC_CASES = [('[4,2,2]', '[4,2,2]'), ('[6,3,3]', '[6,3,3]'),
              ('[7,4,3]', '[5,2,3]'), ('[8,4,4]', '[8,4,4]')]
LDPC_LIFTS = (1, 4, 16, 64, 128, 256, 512)


def mix_tuples(lengths=(5, 6, 7, 8)):
    """All ordered four-tuples (C_0, C_1, C'_0, C'_1) from the catalogue with
    both A-side codes of one length and both B-side codes of another."""
    from .codes import by_length
    bl = by_length()
    out = []
    for n in lengths:
        for a0 in bl.get(n, []):
            for a1 in bl.get(n, []):
                for b0 in bl.get(n, []):
                    for b1 in bl.get(n, []):
                        out.append((a0, a1, b0, b1))
    return out


def base_schedule(Q, restarts=8):
    """A depth-Delta schedule of the base graph, or None."""
    from .qt import balanced_halvings
    for rho in balanced_halvings(Q.nA):
        for sig in balanced_halvings(Q.nB):
            inst = build(Q.pair, Q.nA, Q.nB, np.array(rho), np.array(sig))
            D = inst.Delta()
            if inst.m == 0 or D == 0 or not core.straddles(inst, D):
                continue
            for s in range(restarts):
                for enl in (False, True):
                    sch = simple.run(inst, D, seed=s, enlarge=enl)
                    if sch is not None:
                        core.verify(inst, sch, D)
                        return inst, sch, D
    return None, None, None


def ldpc_row(a, b, L, maxcells=1.2e8):
    """Lift one base pair to |G| = L and verify the full code."""
    from .group import Group
    from .lift import lift_and_verify
    from .qt import QT
    Q = QT(a, b)
    inst, sch, D = base_schedule(Q)
    if sch is None:
        return None
    nq = Q.nA * Q.nB * L
    nchk = sum(int(Q.pair[f][0].shape[0]) * int(Q.pair[f][1].shape[0])
               for f in ('X0', 'X1')) * L
    # HX and HZ are dense, and the propagation check allocates another
    # (n + rX + rZ) x rX bit-matrix, so budget for roughly four of these
    if nq * nchk > maxcells:
        return dict(a=a, b=b, L=L, n=nq, D=D, skipped=True)
    G = Group.cyclic(L)
    out = lift_and_verify(Q.pair, Q.nA, Q.nB, G,
                          [0] * Q.nA, [0] * Q.nB, sch, D)
    code = out['code']
    wr = int(max(code.HX.sum(1).max(initial=0), code.HZ.sum(1).max(initial=0)))
    wc = int((code.HX.sum(0) + code.HZ.sum(0)).max(initial=0))
    r = out['report']
    return dict(a=a, b=b, L=L, n=code.n, base_m=inst.m, D=D,
                delta=code.delta, depth=out['depth'], wr=wr, wc=wc,
                css=out['is_css'],
                ok=bool(r['proper'] and r['contam'] and r['sim']),
                skipped=False)


def main(which, seeds=(0, 1)):
    hist = {}
    if which in ('scale', 'aspect', 'general'):
        cases = CODE_CASES['scale' if which == 'general' else which]
        wide = which == 'aspect'
        print(f"{'nA x nB':>10s} {'squares':>8s} {'|E|':>8s} {'Delta':>6s} "
              f"{'gap':>4s} {'str':>4s} {'side':>6s} {'secs':>7s}  note")
        for (nA, kA, nB, kB) in cases:
            for sd in seeds:
                p = code_pairs(nA, kA, nB, kB, sd, degenerate=wide,
                               identical=(which == 'general'))
                if p is None:
                    continue
                r = attempt(p, nA, nB, sd)
                if r is None:
                    continue
                hist[key(r)] = hist.get(key(r), 0) + 1
                print(f"{f'{nA} x {nB}':>10s} {nA*nB:8d} {r['m']:8d} {r['D']:6d} "
                      f"{str(r['gap']):>4s} {r['nstr']:4d} {r['side']:>6s} "
                      f"{r['secs']:7.1f}  {r['note']}", flush=True)
    elif which == 'rates':
        rng = random.Random(5)
        sides = {}
        for t in range(90):
            nA = rng.randrange(8, 19); nB = rng.randrange(8, 19)
            kA = rng.choice([1, 2, nA // 2, nA - 2, nA - 1])
            kB = rng.choice([1, 2, nB // 2, nB - 2, nB - 1])
            kA = min(max(kA, 1), nA - 1); kB = min(max(kB, 1), nB - 1)
            p = code_pairs(nA, kA, nB, kB, t, degenerate=True)
            if p is None:
                continue
            r = attempt(p, nA, nB, t, tries=24)
            if r is None or r['note'] == 'timeout':
                continue
            hist[key(r)] = hist.get(key(r), 0) + 1
            sides[r['side']] = sides.get(r['side'], 0) + 1
        print('  Delta side:', sides)
    elif which in ('expander', 'exsweep'):
        sweep = which == 'exsweep'
        print(f"{'A (n,c,d)':>12s} {'B (n,c,d)':>12s} {'squares':>8s} {'|E|':>8s} "
              f"{'Delta':>6s} {'gap':>4s} {'str':>4s} {'secs':>7s}  note")
        for case in EXP_CASES[which]:
            args = case + case if sweep else case
            nA, cA, dA, nB, cB, dB = args
            for sd in (range(4) if sweep else seeds):
                p = expander_pairs(nA, cA, dA, nB, cB, dB, sd)
                if p is None:
                    continue
                r = attempt(p, nA, nB, sd, tries=12, seeds=6)
                if r is None:
                    continue
                hist[key(r)] = hist.get(key(r), 0) + 1
                print(f"{f'({nA},{cA},{dA})':>12s} {f'({nB},{cB},{dB})':>12s} "
                      f"{nA*nB:8d} {r['m']:8d} {r['D']:6d} {str(r['gap']):>4s} "
                      f"{r['nstr']:4d} {r['secs']:7.1f}  {r['note']}", flush=True)
    elif which == 'mix':
        from .qt import QT
        print('  four INDEPENDENT local codes per instance (general four-code form).')
        print(f"{'A: C0 / C1':>19s} {'B: C0 / C1':>19s} {'|E|':>7s} {'Delta':>6s} "
              f"{'gap':>4s} {'str':>4s}  note")
        ndiff = 0
        for (a0, a1, b0, b1) in mix_tuples():
            Q = QT.mixed(a0, a1, b0, b1)
            assert commutes(Q.pair)
            ndiff += not Q.is_A
            # the FULL balanced-halving sweep, not `attempt`'s random sample:
            # these instances are small, and a random sample of halvings can
            # report a gap where the full sweep reaches Delta
            inst, sch, D = base_schedule(Q)
            if sch is not None:
                hist[0] = hist.get(0, 0) + 1
                continue
            r = attempt(Q.pair, Q.nA, Q.nB, 0, tries=32)
            if r is None:
                continue
            hist[key(r)] = hist.get(key(r), 0) + 1
            print(f"{a0+' / '+a1:>19s} {b0+' / '+b1:>19s} {r['m']:7d} "
                  f"{r['D']:6d} {str(r['gap']):>4s} {r['nstr']:4d}  {r['note']}",
                  flush=True)
        print(f'  {ndiff} of {len(mix_tuples())} tuples have at least one side mixed')
    elif which == 'ldpc':
        print('  small bounded-weight local codes; size comes from the lift.')
        print('  w_r = max check weight, w_c = max qubit degree, Delta = max(w_r, w_c).')
        print(f"{'base pair':>20s} {'|G|':>5s} {'qubits':>8s} {'base |E|':>9s} "
              f"{'w_r':>4s} {'w_c':>4s} {'Delta':>6s} {'depth':>6s} {'css':>5s}  verified")
        for (a, b) in LDPC_CASES:
            for L in LDPC_LIFTS:
                r = ldpc_row(a, b, L)
                if r is None:
                    print(f'{a+" x "+b:>20s}  no depth-Delta base schedule')
                    break
                if r['skipped']:
                    print(f'{a+" x "+b:>20s} {L:5d} {r["n"]:8d}  (dense lift too large, skipped)')
                    continue
                hist[r['depth'] - r['D']] = hist.get(r['depth'] - r['D'], 0) + 1
                print(f'{a+" x "+b:>20s} {L:5d} {r["n"]:8d} {r["base_m"]:9d} '
                      f'{r["wr"]:4d} {r["wc"]:4d} {r["D"]:6d} {r["depth"]:6d} '
                      f'{str(r["css"]):>5s}  {r["ok"]}', flush=True)
    else:
        raise SystemExit(f"unknown battery {which!r}; pick one of {BATTERIES}")
    print('  gap histogram:', dict(sorted(hist.items(), key=lambda kv: str(kv[0]))))
    return hist


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in BATTERIES:
        raise SystemExit(f"usage: python -m autext.qt.stress {{{'|'.join(BATTERIES)}}}")
    main(sys.argv[1],
         seeds=tuple(int(x) for x in sys.argv[2:]) or (0, 1))
