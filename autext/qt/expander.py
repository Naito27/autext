"""Tanner codes of random (c,d)-regular bipartite graphs.

The local code is the Tanner code of a random (c,d)-regular bipartite graph:
H is the incidence matrix of a configuration-model graph on n variable nodes of
degree c and r = nc/d check nodes of degree d (multi-edges rejected), and the
code is C = ker H, whose generator G is taken as a kernel basis straight from
elimination.  So H is sparse and exactly regular while G is dense and completely
unstructured -- the X families inherit the sparse side and the Z families the
dense side, a strongly lopsided degree profile with no algebraic structure
anywhere.

Generator module only: `autext.qt.stress` draws two of these per side (one per
index c) to build the expander battery.
"""
import numpy as np
from .random_codes import dual_basis


def regular_bipartite(n, c, d, rng, rounds=20000):
    """Incidence matrix of a random (c,d)-regular bipartite graph, simple.
    Configuration model with multi-edges repaired by random half-edge swaps."""
    assert (n * c) % d == 0, 'nc must be divisible by d'
    r = n * c // d
    var = np.repeat(np.arange(n), c)
    chk = np.repeat(np.arange(r), d)
    rng.shuffle(chk)
    for _ in range(rounds):
        key = var * r + chk
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        bad = np.flatnonzero(cnt[inv] > 1)
        if len(bad) == 0:
            break
        a = int(rng.choice(bad)); b = int(rng.integers(len(chk)))
        chk[a], chk[b] = chk[b], chk[a]
    else:
        return None
    H = np.zeros((r, n), int)
    H[chk, var] = 1
    if (H.sum(0) != c).any() or (H.sum(1) != d).any():
        return None
    return H


def expander_code(n, c, d, rng):
    H = regular_bipartite(n, c, d, rng)
    if H is None:
        return None
    G = dual_basis(H)                        # basis of ker H
    if G.shape[0] == 0 or G.sum() == 0:
        return None
    assert (H.dot(G.T) % 2 == 0).all()
    return G, H
