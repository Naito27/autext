"""How large can the true optimum minus Delta actually be?

For instances that are infeasible at Delta, find the exact optimum with CP-SAT
(searching Delta+1, Delta+2, ...).  This is the empirical counterpart of the
a priori bounds: the band certificate allows a gap up to 0.8 Delta, the greedy
blocking-chain theorem allows 3 Delta, but what does the optimum do?
"""
import sys, random
import numpy as np
from .qt import QT
from .core import straddles, layer_hall
from .exact import solve
from .build import best_depth_qt, from_qt

SMALL = ['[2,1,2]', '[3,1,3]', '[4,2,2]', '[5,2,3]', '[6,3,3]', '[7,4,3]']


def optimum(inst, lo, hi, tl=60):
    for T in range(lo, hi + 1):
        st, _ = solve(inst, T, tl=tl, workers=4)
        if st == 'FEASIBLE':
            return T
        if st == 'unknown':
            return None
    return None


def main(per=6, tl=60, seed=11, maxedges=520):
    rng = random.Random(seed)
    hist = {}
    worst = []
    for a in SMALL:
        for b in SMALL:
            qt = QT(a, b); D = qt.Delta()
            # the edge count does not depend on the orientation
            if from_qt(qt, np.zeros(qt.nA, int), np.zeros(qt.nB, int)).m > maxedges:
                continue
            done = 0
            tries = 0
            while done < per and tries < 200:
                tries += 1
                r = np.array([rng.randint(0, 1) for _ in range(qt.nA)])
                s = np.array([rng.randint(0, 1) for _ in range(qt.nB)])
                if best_depth_qt(qt, r, s, span=1, seeds=8) is not None:
                    continue                       # feasible at Delta, gap 0
                # infeasible for the greedy: get the true optimum
                inst = from_qt(qt, r, s)
                opt = optimum(inst, D, D + 4, tl=tl)
                if opt is None:
                    continue
                g = opt - D
                hist[g] = hist.get(g, 0) + 1
                done += 1
                # straddling and layerwise Hall, recorded as diagnostics only
                mo = straddles(inst, D)
                lh, _ = layer_hall(inst, D)
                if g >= 2:
                    worst.append((a, b, tuple(int(u) for u in r),
                                  tuple(int(u) for u in s), D, opt))
                print(f'  {a} x {b}  rho={"".join(map(str,r))} '
                      f'sig={"".join(map(str,s))}  Delta={D:3d}  opt={opt:3d} '
                      f'gap=+{g}  (straddle={int(mo)}, layerHall={int(lh)})')
    print()
    print('  histogram of opt - Delta on orientations where the greedy fails at Delta:')
    print('   ', dict(sorted(hist.items())))
    print(f'  instances with gap >= 2: {len(worst)}')
    for w in worst[:15]:
        print('     ', w)


if __name__ == '__main__':
    main(per=int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         tl=float(sys.argv[2]) if len(sys.argv) > 2 else 60)
