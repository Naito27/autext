"""Build a scheduling instance from a general quantum-Tanner specification.

`pairs` maps each family to its own matrix pair, so the four families may use
FOUR INDEPENDENT matrices -- assumption [A] (H_0=H_1, G_0=G_1, ...) is not
required, and neither is systematic form, complementarity, regularity or
weight-constancy.  Only the halving (rho, sigma) and the canonical sandwich are
used, and those depend on the quadrant alone.
"""
import numpy as np
from .qt import FAMS, roles
from .core import Inst

GROUP = {}
for _q in range(4):
    _E, _M, _L = roles(_q)
    GROUP[_q] = {_E: 1, _M[0]: 2, _M[1]: 2, _L: 3}


def build(pairs, nA, nB, rho, sigma):
    rho = np.asarray(rho); sigma = np.asarray(sigma)
    quad = 2 * rho[:, None] + sigma[None, :]          # nA x nB
    # size the output first and fill it in place: concatenating a list of
    # per-family arrays holds both copies at once, which is the peak on large
    # instances
    tot = 0
    for f in FAMS:
        MA, MB = pairs[f]
        tot += int(MA.sum()) * int(MB.sum())
    sq = np.empty(tot, np.int32); ck = np.empty(tot, np.int64)
    gr = np.empty(tot, np.int8)
    off = 0
    base = 0
    for f in FAMS:
        MA, MB = pairs[f]
        rA, iA = np.nonzero(MA)
        rB, jB = np.nonzero(MB)
        pa, pb = len(rA), len(rB)
        if pa == 0 or pb == 0:
            base += MA.shape[0] * MB.shape[0]
            continue
        ii = np.repeat(iA, pb); ra = np.repeat(rA, pb)
        jj = np.tile(jB, pa); rb = np.tile(rB, pa)
        k = pa * pb
        sq[off:off + k] = ii.astype(np.int64) * nB + jj
        ck[off:off + k] = base + ra.astype(np.int64) * MB.shape[0] + rb
        q = quad[ii, jj]
        gmap = np.array([GROUP[qq][f] for qq in range(4)])
        gr[off:off + k] = gmap[q]
        off += k
        base += MA.shape[0] * MB.shape[0]
    sq = sq[:off]; ck = ck[:off]; gr = gr[:off]
    # Compact the check ids.  np.unique would sort a full copy of `ck`, which is
    # the dominant transient allocation on large instances; the id space is only
    # sum_f (rows of MA)x(rows of MB), a few thousand, so mark-and-cumsum does
    # the same relabelling (new id = number of used ids below it = index in
    # sorted order, exactly what np.unique gives) in space O(#checks).
    seen = np.zeros(base, bool)
    seen[ck] = True
    nck = int(seen.sum())
    relabel = np.cumsum(seen) - 1
    ck = relabel[ck]
    return Inst(sq, ck, gr, nA * nB, nck)


def from_qt(qt, rho, sigma):
    return build(qt.pair, qt.nA, qt.nB, rho, sigma)


def best_depth_qt(qt, rho, sigma, span=4, seeds=16, alg=None):
    """Smallest depth in [Delta, Delta+span) that `alg` reaches on QT `qt` at
    the halving (rho, sigma), or None.  A qt-level convenience wrapper around
    the Inst-level algorithms, so scripts that speak the QT language do not
    have to build the instance themselves."""
    from . import simple
    if alg is None:
        alg = simple
    inst = from_qt(qt, rho, sigma)
    D = inst.Delta()
    for k in range(span):
        for sd in range(seeds):
            sch = alg.run(inst, D + k, seed=sd)
            if sch is not None:
                from .core import verify
                verify(inst, sch, D + k)
                return D + k
    return None


if __name__ == '__main__':
    import time
    from .qt import QT
    from .codes import P_OF
    from . import core, simple
    print('=== the layer sweep on the 64 base pairs (systematic halving) ===')
    n = 0; t = 0.0
    for a in list(P_OF):
        for b in list(P_OF):
            qt = QT(a, b)
            k, kp = qt.nA // 2, qt.nB // 2
            rho = np.array([1] * (qt.nA - k) + [0] * k)
            sigma = np.array([0] * kp + [1] * (qt.nB - kp))
            inst = from_qt(qt, rho, sigma)
            D = inst.Delta()
            assert D == qt.Delta(), (a, b, D, qt.Delta())
            t0 = time.time()
            for sd in range(16):
                sch = simple.run(inst, D, seed=sd)
                if sch is not None:
                    core.verify(inst, sch, D); n += 1; break
            t += time.time() - t0
    print(f'  layer sweep {n:2d}/64 at Delta  ({t:5.1f}s)')
