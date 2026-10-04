"""Quadripartite quantum Tanner base graph: the family matrix pairs and degrees.

Eq. (S45) builds the base code from the four classical codes C_0, C_1 (A side)
and C'_0, C'_1 (B side):

    X_0 = H_0 (x) G'_0    X_1 = H_1 (x) G'_1
    Z_0 = G_0 (x) H'_1    Z_1 = G_1 (x) H'_0

Commutation.  Each block of HX HZ^T is a Kronecker product, so it vanishes as
soon as EITHER factor does; the diagonal choice -- H_c G_c^T = 0 on the A side
and G'_c H'_c^T = 0 on the B side, each within a single index c -- kills all
four at once.  So the code is CSS as soon as each (G_c, H_c) is a generator /
parity-check pair of one code, and nothing needs to couple index 0 to index 1.
(That is sufficient, not necessary: `commutes` keeps the full disjunctions.)

**The permutation.**  Taking C_0 = C_1 = C does NOT mean the four families share
matrices: the two copies carry independent labellings of the coordinate set, so
index 1 sees C through a relabelling pi.  The freedom that survives commutation
is one permutation pi_A applied to (H_1, G_1) JOINTLY and one pi_B applied to
(G'_1, H'_1) jointly -- two coupled permutations, giving

    X_0 = (H,        G')       X_1 = (H pi_A,  G' pi_B)
    Z_0 = (G,        H' pi_B)  Z_1 = (G pi_A,  H'      )

`pi_A = pi_B = id` is assumption **[A]**, a strictly special case: it makes
deg_{X_0} = deg_{X_1} and deg_{Z_0} = deg_{Z_1} everywhere.  It is the default
here; pass `piA`/`piB` (or use `QT.random`) for the general construction, and
see `QT.is_A`.  Four INDEPENDENT permutations, one per family, do not commute
in general and do not give a QT code.

Quadrant index q = 2*rho(i) + sigma(j), i.e. 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1),
so that X_0, X_1 are middle in quadrants 0 and 3.
"""
import numpy as np
import itertools
from .codes import get_code

FAMS = ['X0', 'X1', 'Z0', 'Z1']

# canonical orientation, generated from the two halving bits:
#   row bit a controls the pairs (X0,Z1) and (X1,Z0)
#   col bit b controls the pairs (X0,Z0) and (X1,Z1)
def precedences(a, b):
    """Ordered pairs (f_first, f_second) holding at a square with bits (a,b)."""
    p = []
    p += [('Z1', 'X0'), ('X1', 'Z0')] if a == 0 else [('X0', 'Z1'), ('Z0', 'X1')]
    p += [('X0', 'Z0'), ('Z1', 'X1')] if b == 0 else [('Z0', 'X0'), ('X1', 'Z1')]
    return p


def roles(q):
    """(early, (middle1, middle2), late) for quadrant q, from the sandwich."""
    a, b = q // 2, q % 2
    prec = precedences(a, b)
    before = {f: 0 for f in FAMS}
    for f1, f2 in prec:
        before[f2] += 1
    E = [f for f in FAMS if before[f] == 0]
    L = [f for f in FAMS if before[f] == 2]
    M = [f for f in FAMS if before[f] == 1]
    assert len(E) == 1 and len(L) == 1 and len(M) == 2, (q, before)
    return E[0], tuple(M), L[0]


def pairs_from_codes(A0, A1, B0, B1):
    """The Eq. (S45) family layout from four classical code pairs.

    `A_c = (G_c, H_c)` is the A-side code of index `c` and `B_c = (G'_c, H'_c)`
    the B-side one; index 0 and index 1 are INDEPENDENT codes on the same
    coordinate set.  This is the general construction -- `QT` is the special
    case where the two indices are one code seen through a relabelling.

        X0 = (H0, G'0)   X1 = (H1, G'1)   Z0 = (G0, H'1)   Z1 = (G1, H'0)
    """
    (G0, H0), (G1, H1) = A0, A1
    (Gp0, Hp0), (Gp1, Hp1) = B0, B1
    return {'X0': (H0, Gp0), 'X1': (H1, Gp1),
            'Z0': (G0, Hp1), 'Z1': (G1, Hp0)}


def commutes(pairs):
    """Is `pairs` a CSS code, i.e. HX HZ^T = 0 for the base code of Eq. (S45)?

    Exact, and computed on the small factor matrices -- never by forming the
    Kronecker products.  The four blocks of HX HZ^T are

        (H0 G0^T) x (G'0 H'1^T)     (H0 G1^T) x (G'0 H'0^T)
        (H1 G0^T) x (G'1 H'1^T)     (H1 G1^T) x (G'1 H'0^T)

    and `A x B = 0` iff `A = 0` OR `B = 0`, so each block gives a disjunction.
    The *guaranteed* way to satisfy all four is the diagonal one -- H_c G_c^T = 0
    and G'_c H'_c^T = 0, each a condition within a single index c, which is why
    `pairs_from_codes` is CSS for any four code pairs.  A block can also vanish
    through its other factor, so the disjunctions are kept: testing only the
    diagonal would report false negatives.

    Use this on any hand-built `pairs` dict.  Permuting the four families'
    matrices independently breaks commutation, because the relabelling of `H_c`
    must match `G_c`'s.
    """
    H0, Gp0 = pairs['X0']
    H1, Gp1 = pairs['X1']
    G0, Hp1 = pairs['Z0']
    G1, Hp0 = pairs['Z1']

    def z(P, Q):
        return not (P.dot(Q.T) % 2).any()

    return bool((z(H0, G0) or z(Gp0, Hp1))
                and (z(H0, G1) or z(Gp0, Hp0))
                and (z(H1, G0) or z(Gp1, Hp1))
                and (z(H1, G1) or z(Gp1, Hp0)))


def _as_perm(pi, n, side):
    if pi is None:
        return np.arange(n)
    pi = np.asarray(pi, int)
    assert pi.shape == (n,) and np.array_equal(np.sort(pi), np.arange(n)),         f'pi_{side} is not a permutation of {n} coordinates'
    return pi


class QT:
    """The base code of Eq. (S45) from two classical codes and the coupled
    relabelling (pi_A, pi_B).  `piA=piB=None` is assumption [A]."""

    def __init__(self, nameA, nameB, piA=None, piB=None,
                 nameA1=None, nameB1=None):
        """`nameA`/`nameB` are the index-0 codes.  `nameA1`/`nameB1` default to
        them (so index 1 is the same code through the relabelling pi), or name a
        DIFFERENT code of the same length -- Leverrier, Rozendaal and Zemor
        (arXiv:2512.20532) take all four of C_0, C_1, C'_0, C'_1
        independently."""
        self.nameA, self.nameB = nameA, nameB
        self.nameA1 = nameA1 or nameA
        self.nameB1 = nameB1 or nameB
        self.G, self.H = get_code(nameA)      # A side, index 0: (G_0, H_0)
        self.Gp, self.Hp = get_code(nameB)    # B side, index 0: (G'_0, H'_0)
        G1, H1 = get_code(self.nameA1)        # A side, index 1
        Gp1, Hp1 = get_code(self.nameB1)      # B side, index 1
        self.nA, self.nB = self.H.shape[1], self.Hp.shape[1]
        if H1.shape[1] != self.nA or Hp1.shape[1] != self.nB:
            raise ValueError('the two codes on a side must have the same length')
        self.piA = _as_perm(piA, self.nA, 'A')
        self.piB = _as_perm(piB, self.nB, 'B')
        # index 1 sees its side through the relabelling; the pair (H_c, G_c)
        # moves together, which is what keeps HX HZ^T = 0
        self.pair = pairs_from_codes(
            (self.G, self.H), (G1[:, self.piA], H1[:, self.piA]),
            (self.Gp, self.Hp), (Gp1[:, self.piB], Hp1[:, self.piB]))

    @property
    def is_A(self):
        """True iff C_0 = C_1, C'_0 = C'_1 and pi_A = pi_B = id -- i.e. this
        instance satisfies [A]."""
        return bool(self.nameA1 == self.nameA and self.nameB1 == self.nameB
                    and np.array_equal(self.piA, np.arange(self.nA))
                    and np.array_equal(self.piB, np.arange(self.nB)))

    @classmethod
    def mixed(cls, a0, a1, b0, b1, piA=None, piB=None):
        """Four independent local codes, the general form of Leverrier et al."""
        return cls(a0, b0, piA, piB, nameA1=a1, nameB1=b1)

    def __repr__(self):
        a = self.nameA if self.nameA1 == self.nameA else f'{self.nameA}/{self.nameA1}'
        b = self.nameB if self.nameB1 == self.nameB else f'{self.nameB}/{self.nameB1}'
        return f'QT({a} x {b}{"" if self.is_A else ", pi != id"})'

    @classmethod
    def random(cls, nameA, nameB, rng):
        """A QT code on the same two classical codes with random pi_A, pi_B."""
        nA = get_code(nameA)[1].shape[1]
        nB = get_code(nameB)[1].shape[1]
        return cls(nameA, nameB, rng.permutation(nA), rng.permutation(nB))

    # ---------- degrees, independent of any halving ----------
    def deg_f(self, f):
        MA, MB = self.pair[f]
        return np.outer(MA.sum(0), MB.sum(0))          # nA x nB

    def deg_square(self):
        return sum(self.deg_f(f) for f in FAMS)

    def max_check_deg(self):
        best = 0
        for f in FAMS:
            MA, MB = self.pair[f]
            best = max(best, int(MA.sum(1).max() * MB.sum(1).max()))
        return best

    def Delta(self):
        return max(int(self.deg_square().max()), self.max_check_deg())

    # ---------- explicit base graph (for verification / solving) ----------
    def edges(self, rho, sigma):
        """List of (fam, rA, rB, i, j)."""
        E = []
        for f in FAMS:
            MA, MB = self.pair[f]
            for rA in range(MA.shape[0]):
                for i in np.flatnonzero(MA[rA]):
                    for rB in range(MB.shape[0]):
                        for j in np.flatnonzero(MB[rB]):
                            E.append((f, rA, int(rB), int(i), int(j)))
        return E


# ---------- halvings ----------
def balanced_halvings(n):
    """All balanced 0/1 halvings of [n], up to global complement."""
    out = []
    for k in [n // 2]:
        for S in itertools.combinations(range(n), k):
            if n % 2 == 0 and 0 not in S:
                continue                      # kill the complement duplicate
            v = np.zeros(n, int); v[list(S)] = 1
            out.append(v)
    return out


def weight_class_halving(M):
    """Split columns by column-weight class, balanced as far as possible."""
    cw = M.sum(0)
    order = np.argsort(cw, kind='stable')
    v = np.zeros(len(cw), int)
    v[order[len(cw) // 2:]] = 1
    return v
