"""Closed form for the band bound, and its minimum over orientations.

The band theorem certifies depth  Delta_1 + Delta_2 + Delta_3  with no search,
where Delta_g = max( max_v n_g(v), max_c m_g(c) ).  Both maxima have closed
forms in the column and row weights, so the bound -- and hence a CERTIFIED upper
bound on the gap -- can be evaluated for a whole orientation sweep.

Square side.  With  x(i,j) = cw_H(i) cw_{G'}(j)  and  z(i,j) = cw_G(i) cw_{H'}(j),
assumption [A] gives deg_{X_0}=deg_{X_1}=x and deg_{Z_0}=deg_{Z_1}=z, so at a
square in quadrant q,

    q in {0,3} (rho(i)=sigma(j)):  n_1 = n_3 = z,   n_2 = 2x,
    q in {1,2} (rho(i)!=sigma(j)): n_1 = n_3 = x,   n_2 = 2z,

and in particular n_1(v) = n_3(v) and d(v) = 2(x+z) at every square.  Hence
   A_1 = A_3 = max( max_{rho=sigma} z , max_{rho!=sigma} x ),
   A_2   = 2 max( max_{rho=sigma} x , max_{rho!=sigma} z ).

Check side.  A check c = (f,rA,rB) has neighbourhood R_A x R_B; write
p = |R_A|, q = |R_B| and, with (aE,bE) the early quadrant of f,
a = |R_A n rho^{-1}(aE)|, b = |R_B n sigma^{-1}(bE)|.  Then

    m_1(c) = a b,     m_3(c) = (p-a)(q-b),     m_2(c) = pq - m_1 - m_3 .

COROLLARY (a certified gap bound, no search).
    optimum  <=  Delta_1 + Delta_2 + Delta_3 ,
so the gap over Delta is at most (Delta_1+Delta_2+Delta_3) - Delta, minimised
over orientations.  In particular, if every square has 4 n_1(v) <= Delta and
2 n_2(v) <= Delta, and every check has 4 m_1(c) <= Delta, 2 m_2(c) <= Delta and
4 m_3(c) <= Delta -- i.e. every rectangle is split near evenly by the halving --
then the bound is exactly Delta and the greedy has nothing to beat.
"""
import itertools, random
import numpy as np
from .qt import QT, FAMS, roles

# family -> the (unique) quadrant in which it is early, read straight off the
# sandwich roles
EARLY_Q = {roles(q)[0]: q for q in range(4)}

# early quadrant of each family, as (a,b)
EQ = {f: (EARLY_Q[f] // 2, EARLY_Q[f] % 2) for f in FAMS}


class Bands:
    """The [A]-only closed form.  The square-side derivation above needs
    deg_{X_0} = deg_{X_1} = x and deg_{Z_0} = deg_{Z_1} = z, which holds exactly
    when pi_A = pi_B = id; with a non-trivial relabelling the four families have
    four different degree grids, so ``self.x``/``self.z`` would describe the
    wrong graph.  The constructor therefore asserts ``qt.is_A`` rather than
    returning a bound for a graph it does not describe; ``band_bound(inst)``
    covers the general case."""

    def __init__(self, qt):
        assert qt.is_A, ('Bands is the [A] closed form and needs pi = id; '
                         'use band_bound(inst) for the general case')
        self.qt = qt
        self.x = np.outer(qt.H.sum(0), qt.Gp.sum(0))      # nA x nB
        self.z = np.outer(qt.G.sum(0), qt.Hp.sum(0))
        self.rows = {}
        for f in FAMS:
            MA, MB = qt.pair[f]
            self.rows[f] = (MA, MB, MA.sum(1), MB.sum(1))

    def bound(self, rho, sigma):
        same = (rho[:, None] == sigma[None, :])
        A13 = max(self.z[same].max(initial=0), self.x[~same].max(initial=0))
        A2 = 2 * max(self.x[same].max(initial=0), self.z[~same].max(initial=0))
        B1 = B2 = B3 = 0
        for f in FAMS:
            MA, MB, p, q = self.rows[f]
            aE, bE = EQ[f]
            a = MA.dot((rho == aE).astype(int))
            b = MB.dot((sigma == bE).astype(int))
            m1 = np.outer(a, b)
            m3 = np.outer(p - a, q - b)
            m2 = np.outer(p, q) - m1 - m3
            B1 = max(B1, int(m1.max(initial=0)))
            B2 = max(B2, int(m2.max(initial=0)))
            B3 = max(B3, int(m3.max(initial=0)))
        return int(max(A13, B1) + max(A2, B2) + max(A13, B3))


def sweep(qt, nsample=4000, seed=0):
    B = Bands(qt)
    rng = random.Random(seed)
    best = None
    cands = []
    if qt.nA <= 8 and qt.nB <= 8 and 2 ** (qt.nA + qt.nB) <= 20000:
        for r in itertools.product([0, 1], repeat=qt.nA):
            for s in itertools.product([0, 1], repeat=qt.nB):
                cands.append((np.array(r), np.array(s)))
    else:
        k, kp = qt.nA // 2, qt.nB // 2
        cands.append((np.array([1] * (qt.nA - k) + [0] * k),
                      np.array([0] * kp + [1] * (qt.nB - kp))))
        for _ in range(nsample):
            cands.append((np.array([rng.randint(0, 1) for _ in range(qt.nA)]),
                          np.array([rng.randint(0, 1) for _ in range(qt.nB)])))
    for r, s in cands:
        v = B.bound(r, s)
        if best is None or v < best[0]:
            best = (v, r.copy(), s.copy())
    return best


def band_bound(inst):
    """Direct count of the band certificate: Delta_i is the maximum degree of
    the group-i subgraph over squares AND checks, and depth Delta_1+Delta_2+
    Delta_3 is achievable (colour each group-i subgraph with Delta_i colours by
    Koenig and lay the three colour sets out in consecutive bands).  Unlike the
    closed form in `Bands.bound` it assumes nothing about the four families, so
    it applies at pi != id as well."""
    ngC = np.bincount(inst.ck.astype(np.int64) * 4 + inst.grp,
                      minlength=inst.nck * 4).reshape(inst.nck, 4)
    out = [max(int(inst.ng[:, g].max(initial=0)), int(ngC[:, g].max(initial=0)))
           for g in (1, 2, 3)]
    return sum(out), tuple(out)


if __name__ == '__main__':
    from .codes import P_OF
    from .build import from_qt
    NAMES = list(P_OF)

    print('=== sanity: closed form vs. the direct count ===')
    rng = random.Random(0); bad = 0; n = 0
    for a in NAMES[:5]:
        for b in NAMES[:5]:
            qt = QT(a, b); B = Bands(qt)
            for _ in range(12):
                r = np.array([rng.randint(0, 1) for _ in range(qt.nA)])
                s = np.array([rng.randint(0, 1) for _ in range(qt.nB)])
                n += 1
                bad += (B.bound(r, s) != band_bound(from_qt(qt, r, s))[0])
    print(f'  agree on {n-bad}/{n}')

    print()
    print('=== certified gap: min over orientations of (bands - Delta) ===')
    print(f"{'pair':>22s} {'Delta':>6s} {'bands@sys':>10s} {'min bands':>10s} "
          f"{'certified gap':>14s}")
    worst = 0
    for a in NAMES:
        for b in NAMES:
            qt = QT(a, b); D = qt.Delta(); B = Bands(qt)
            k, kp = qt.nA // 2, qt.nB // 2
            rho = np.array([1] * (qt.nA - k) + [0] * k)
            sigma = np.array([0] * kp + [1] * (qt.nB - kp))
            at_sys = B.bound(rho, sigma)
            best, r, s = sweep(qt)
            worst = max(worst, best - D)
            print(f'{a+" x "+b:>22s} {D:6d} {at_sys:10d} {best:10d} '
                  f'{best-D:+14d}')
    print(f'  worst certified gap over the 64 pairs: +{worst}')
