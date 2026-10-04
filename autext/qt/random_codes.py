"""Random classical local codes: no systematic form, no complementarity, no
structural assumption of any kind.

`G` is a random full-rank k x n matrix over F2 and `H` a basis of its dual, both
kept as they come out of the elimination -- so H is NOT [P^T|I] and G is NOT
[I|P] -- and the columns are then scrambled.  These are the generators behind
every "general" row of the stress battery, and they are what makes assumption
[A] (identical parity-check and generator matrices across the four families)
false by construction.

Exports: rref, dual_basis, random_code, random_code_degenerate.
"""
import numpy as np

def rref(M):
    M = M.copy() % 2
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        pr = None
        for i in range(r, rows):
            if M[i, c]:
                pr = i; break
        if pr is None:
            continue
        M[[r, pr]] = M[[pr, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c); r += 1
        if r == rows:
            break
    return M[:r], piv


def dual_basis(G):
    """A basis of { x : G x^T = 0 } as rows of H."""
    R, piv = rref(G)
    k, n = R.shape
    free = [c for c in range(n) if c not in piv]
    H = []
    for f in free:
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = R[i, f]
        H.append(v % 2)
    return np.array(H, dtype=int) if H else np.zeros((0, n), int)


def random_code(rng, n, k, tries=200):
    for _ in range(tries):
        G = np.array([[rng.randint(0, 1) for _ in range(n)] for _ in range(k)])
        R, piv = rref(G)
        if R.shape[0] != k:
            continue
        if (G.sum(0) == 0).any():
            continue                       # a zero column kills a whole family
        H = dual_basis(G)
        if H.shape[0] != n - k or (H.sum(0) == 0).any():
            continue
        if (H.dot(G.T) % 2 != 0).any():
            continue
        # scramble so nothing is in systematic position
        P = np.eye(n, dtype=int)[rng.sample(range(n), n)]
        return (G.dot(P) % 2), (H.dot(P) % 2)
    return None


def random_code_degenerate(rng, n, k, tries=400):
    """`random_code` without the two cosmetic restrictions: zero-weight columns
    are allowed and the columns are left where elimination put them.  A
    degenerate column is legitimate input -- it leaves one family with no edge
    at that coordinate -- so the construction must tolerate it.  This is the
    generator behind the aspect-ratio and extreme-rate batteries, where thin
    sides make degenerate columns common."""
    for _ in range(tries):
        G = np.array([[rng.randint(0, 1) for _ in range(n)] for _ in range(k)])
        R, _ = rref(G)
        if R.shape[0] != k:
            continue
        H = dual_basis(G)
        if H.shape[0] != n - k or (H.dot(G.T) % 2 != 0).any():
            continue
        if H.sum() == 0 or G.sum() == 0:
            continue
        return G, H
    return None
