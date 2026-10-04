"""Which local codes give a qLDPC quantum Tanner code -- the Leverrier et al.
criterion.

Leverrier, Rozendaal and Zemor (arXiv:2512.20532) select the local codes for
hardware relevance as follows.  The support of a stabiliser generator is a
product of two local codewords, so its weight obeys

    w >= max(d_0^perp d'_0, d_1^perp d'_1, d_0 d'^perp_1, d_1 d'^perp_0),

and they "focus here on local codes that give reasonably small weights, namely
C_i with parameters [6,3,3] or [8,4,4], and C'_i with parameters [2,1,2] or
[6,3,3].  **All these codes have dual codes with identical parameters.**"  They
also discard the all-repetition choice: "One could use only repetition codes
[2,1,2], thus getting ... weight 4, but these would not beat the (rotated) toric
code".

Two conditions, then:

1. **the dual has identical parameters** -- `[n, k, d]` with `[n, n-k, d^perp]`
   and `k = n-k`, `d = d^perp`.  This is what keeps BOTH sides sparse: the four
   families use `(H, G')` and `(G, H')`, so a code with a dense dual makes the
   Z-families dense and the code is not LDPC however small the X-families are.
2. **generator weight `w >= 6`**, discarding the weight-4 (toric-like) corner.

Applied to `codes.ALL_P` the first condition selects exactly
`[2,1,2], [4,2,2], [6,3,3], [8,4,4]` -- the three names Leverrier et al. use,
plus `[4,2,2]`, which they do not (it has the same row weight as `[2,1,2]`, so
it only reproduces the same weights).
"""
import itertools

import numpy as np

from .codes import ALL_P, get_code, min_distance
from ..linalg import rref_gf2

__all__ = ["self_dual_parameters", "eligible_codes", "generator_weight",
           "perm_classes", "qldpc_pairs", "REF36_PAIRS"]

# the three combinations Leverrier et al. instantiate (weights 6, 8, 9)
REF36_PAIRS = (('[6,3,3]', '[2,1,2]'), ('[8,4,4]', '[2,1,2]'),
               ('[6,3,3]', '[6,3,3]'))


def self_dual_parameters(name):
    """Does this code have a dual with identical parameters?"""
    G, H = get_code(name)
    n, k = G.shape[1], G.shape[0]
    return bool(H.shape[0] == k and min_distance(G) == min_distance(H))


def eligible_codes():
    """Catalogue names passing condition 1, ordered by length."""
    out = [nm for nm in ALL_P if self_dual_parameters(nm)]
    return sorted(out, key=lambda nm: get_code(nm)[0].shape[1])


def generator_weight(a, b):
    """`w` for the pair: the stabiliser support is a product of two codewords,
    so its weight is the product of the two row weights."""
    _, Ha = get_code(a)
    Gb, _ = get_code(b)
    return int(Ha.sum(1).max()) * int(Gb.sum(1).max())


def perm_classes(name):
    """One representative column permutation per DISTINCT code.

    Leverrier et al.: "there are 30 distinct possible codes that are obtained
    from the canonical parity-check matrices ... through a column permutation.
    Since the parameters of the quantum Tanner code do not change under a global
    permutation of the qubits, it is in fact sufficient to pick H_0 or H'_0 of
    the form above, and only try the 30 permutations for H_1 and H'_1."  So the
    permutation is applied to index 1 only -- exactly `QT(a, b, piA, piB)`.

    Note that this deduplicates by CODE, which is the right invariant for the
    code parameters but not for the base Tanner graph: two permutations giving
    the same code can give row-inequivalent matrices, hence different
    scheduling instances.
    """
    _, H = get_code(name)
    n = H.shape[1]
    reps = {}
    for p in itertools.permutations(range(n)):
        reps.setdefault(rref_gf2(H[:, list(p)])[0].tobytes(), list(p))
    return list(reps.values())


def qldpc_pairs(wmin=6, wmax=12):
    """`(A-code, B-code, w)` for every eligible pair, weight-filtered.

    Leverrier et al. keep `w` in {6, 8, 9} -- "we seek to minimize this value
    for hardware implementations".  The default window here stretches that to
    12 to cover the two mixed `[6,3,3]`/`[8,4,4]` pairings, but stops short of
    `[8,4,4] x [8,4,4]`: its `w = 16` and `Delta = 20` put it outside the regime
    the criterion is selecting for, and its base graph (1024 edges) is far
    larger than any other eligible pair's.  Pass `wmax=None` to include it.
    """
    el = eligible_codes()
    out = []
    for a in el:
        for b in el:
            w = generator_weight(a, b)
            if w < wmin or (wmax is not None and w > wmax):
                continue
            out.append((a, b, w))
    return sorted(out, key=lambda t: (t[2], t[0], t[1]))
