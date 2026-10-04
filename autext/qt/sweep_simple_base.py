"""The layer sweep on every base pair of the catalogue, swept over all balanced
halvings.

    python -m autext.qt.sweep_simple_base            # pi = id, assumption [A]
    python -m autext.qt.sweep_simple_base <seed> ... # one sweep per pi seed

With no argument the four families share matrices (pi_A = pi_B = id).  With
seeds, each sweep draws a fresh coupled relabelling (`QT.random`) -- the general
construction of Eq. (S45) with C_0 = C_1.  Every greedy miss is then handed to
CP-SAT, so a reported failure is a certified obstruction and not solver
weakness; halvings failing `straddles` are skipped, not counted.
"""
import sys
import numpy as np
from . import codes, core, simple
from .build import build
from .qt import QT, balanced_halvings


def sweep(Q, restarts=6, span=3):
    """Smallest gap over Delta the layer sweep reaches on any balanced halving,
    and the halvings that passed `straddles`."""
    best, D, straddling = None, 0, []
    for rho in balanced_halvings(Q.nA):
        for sig in balanced_halvings(Q.nB):
            inst = build(Q.pair, Q.nA, Q.nB, np.array(rho), np.array(sig))
            D = inst.Delta()
            if inst.m == 0 or D == 0 or not core.straddles(inst, D):
                continue
            straddling.append((rho, sig))
            for k in range(span):
                if best is not None and k >= best:
                    break
                for s in range(restarts):
                    sch = simple.run(inst, D + k, seed=s)
                    if sch is not None:
                        core.verify(inst, sch, D + k)
                        best = k if best is None else min(best, k)
                        break
                if best == 0:
                    break
            if best == 0:
                break
        if best == 0:
            break
    return best, D, straddling


def adjudicate(Q, straddling, D):
    """The greedy missed depth D: is that an obstruction or is it the greedy?"""
    from .exact import solve
    for rho, sig in straddling:
        inst = build(Q.pair, Q.nA, Q.nB, np.array(rho), np.array(sig))
        if solve(inst, D, tl=30, workers=4)[0] == 'FEASIBLE':
            return 'greedy weakness'
    return 'certified INFEASIBLE at Delta'


def main(seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    hist = {}
    print(f"{'pair':>20s} {'Delta':>6s} {'gap':>4s}  note")
    for a in codes.P_OF:
        for b in codes.P_OF:
            Q = QT.random(a, b, rng) if rng is not None else QT(a, b)
            best, D, straddling = sweep(Q)
            note = ''
            if best != 0:
                note = ('no straddling halving' if not straddling
                        else adjudicate(Q, straddling, D))
                if best is not None:
                    note += f' (greedy reached Delta+{best})'
            # key by REASON when there is no gap: `best is None` covers three
            # unrelated outcomes -- no straddling halving (depth Delta provably
            # impossible), a certified obstruction, and plain greedy weakness --
            # which are the distinctions the adjudication above draws
            k = best if best == 0 else (note or best)
            hist[k] = hist.get(k, 0) + 1
            print(f"{a+' x '+b:>20s} {D:6d} {str(best):>4s}  {note}", flush=True)
    print('\n  gap histogram:',
          dict(sorted(hist.items(), key=lambda kv: str(kv[0]))))
    return hist


if __name__ == '__main__':
    seeds = [int(x) for x in sys.argv[1:]] or [None]
    for sd in seeds:
        print(f"=== pi {'= id (assumption [A])' if sd is None else f'seed {sd}'} ===")
        main(sd)
