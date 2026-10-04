"""Classical base codes for the QT construction.

Convention:
    G = [I_k | P]   (generator, k x n)
    H = [P^T | I_r] (parity check, r x n),  r = n-k
so that H G^T = P^T + P^T = 0 over F_2.
"""
import numpy as np
import itertools

P_OF = {
    '[2,1,2]': [[1]],
    '[3,1,3]': [[1, 1]],
    '[4,2,2]': [[1, 0], [0, 1]],
    '[5,1,5]': [[1, 1, 1, 1]],
    '[5,2,3]': [[1, 1, 0], [0, 1, 1]],
    '[6,3,3]': [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    '[7,4,3]': [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
    '[8,4,4]': [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
}


# Extra codes, one per (length, dimension) not already in P_OF.  These exist so
# the four local codes of Eq. (S45) can be genuinely DIFFERENT -- Leverrier,
# Rozendaal and Zemor (arXiv:2512.20532) take C_0, C_1, C'_0, C'_1
# independently, and a side can only mix two codes of the same length.  Kept
# separate from P_OF so that the 64 base pairs are unchanged.  Distances are
# computed, not asserted.
EXTRA_P = {
    '[3,2,2]': [[1], [1]],
    '[4,1,4]': [[1, 1, 1]],
    '[4,3,2]': [[1], [1], [1]],
    '[5,3,2]': [[0, 1], [1, 1], [0, 1]],
    '[6,2,4]': [[0, 1, 1, 1], [1, 0, 1, 1]],
    '[6,4,2]': [[1, 1], [1, 0], [1, 1], [1, 1]],
    '[7,3,4]': [[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
    '[7,2,4]': [[0, 1, 1, 0, 1], [1, 1, 1, 1, 1]],
    '[8,3,4]': [[1, 0, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 0, 1, 1]],
    '[8,2,5]': [[1, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 1]],
}

ALL_P = {**P_OF, **EXTRA_P}


def by_length():
    """length -> the catalogue names of that length, for mix-and-match."""
    out = {}
    for name, P in ALL_P.items():
        k, r = len(P), len(P[0])
        out.setdefault(k + r, []).append(name)
    return out


def systematic(P):
    P = np.array(P, dtype=int)
    k, r = P.shape
    G = np.hstack([np.eye(k, dtype=int), P])
    H = np.hstack([P.T, np.eye(r, dtype=int)])
    return G, H


def min_distance(G):
    k, n = G.shape
    best = n + 1
    for coeffs in itertools.product([0, 1], repeat=k):
        if not any(coeffs):
            continue
        w = int(np.array(coeffs).dot(G).__mod__(2).sum())
        best = min(best, w)
    return best


def get_code(name):
    """Return (G, H): rows of G span the code, H is its parity check.

    This is Leverrier et al.'s convention -- C_i = ker H_i and the rows of
    G_i span C_i, so H_i G_i^T = 0, which is what makes Eq. (S45) commute.
    """
    G, H = systematic(ALL_P[name])
    assert (H.dot(G.T) % 2 == 0).all(), f'{name}: H G^T != 0'
    return G, H


def code_report(name):
    G, H = get_code(name)
    k, n = G.shape
    d = min_distance(G)
    return dict(
        name=name, n=n, k=k, d=d,
        cw_G=tuple(G.sum(0)), cw_H=tuple(H.sum(0)),
        rw_G=tuple(G.sum(1)), rw_H=tuple(H.sum(1)),
        complementary=len(set(G.sum(0) + H.sum(0))) == 1,
    )


def overlap_pairs(G, H):
    """Edges of the overlap graph on the column set: the size-two sets
    S = supp(G[a]) cap supp(H[b]), returned as sorted column-index tuples."""
    out = []
    for a in range(G.shape[0]):
        for b in range(H.shape[0]):
            S = tuple(np.flatnonzero(G[a] & H[b]).tolist())
            if len(S) == 2:
                out.append(S)
    return sorted(set(out))


if __name__ == '__main__':
    print(f"{'code':10s} {'n,k,d':10s} {'cw_H':26s} {'cw_G':26s} {'compl':6s}")
    print('-' * 84)
    for name in P_OF:
        r = code_report(name)
        print(f"{r['name']:10s} [{r['n']},{r['k']},{r['d']}]".ljust(21),
              f"{str(r['cw_H']):26s} {str(r['cw_G']):26s} {str(r['complementary']):6s}")

    print()
    print('=== FINGERPRINT 1: [6,3,3] size-two overlaps (expected: the 6-cycle')
    print('    (0,4),(0,5),(1,3),(1,5),(2,3),(2,4) ) ===')
    G, H = get_code('[6,3,3]')
    got = overlap_pairs(G, H)
    expected = [(0, 4), (0, 5), (1, 3), (1, 5), (2, 3), (2, 4)]
    print('  got     :', got)
    print('  expected:', expected)
    print('  MATCH   :', sorted(got) == sorted(expected))

    print()
    print('=== FINGERPRINT 2: [7,4,3] column weights of H (expected [1,1,1,2,2,2,3]) ===')
    r = code_report('[7,4,3]')
    print('  got (multiset)     :', sorted(r['cw_H']))
    print('  expected (multiset):', [1, 1, 1, 2, 2, 2, 3])
    print('  MATCH              :', sorted(r['cw_H']) == [1, 1, 1, 2, 2, 2, 3])
    print('  complementary?     :', r['complementary'], '(expected False for [7,4,3])')
