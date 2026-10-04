"""The analytic sandwich schedule of Eq. (S24) for BB codes.

For a BB code ``H_X = [A | B]``, ``H_Z = [B^T | A^T]`` with ``A``, ``B`` sums of
``d_A``, ``d_B`` distinct monomials, each monomial is a permutation matrix, so
the edge colouring the SM asks for is free: **the monomial index is the colour**.
Following Fig. S1's naming, with ``L`` the first block of qubits and ``R`` the
second,

    X-L : a_ij      X-R : b_kl      Z-R : a*_ij     Z-L : b*_kl

``P_box`` merges the A-edges with their conjugates into the **middle** band, and
splits the B-edges between the **early** and **late** bands: ``ceil(d_B/2)``
colour classes of ``b`` go early and the rest late, with ``b*`` taking the
opposite band (SM Sec. S2). An ``a``-edge and its conjugate ``a*`` share a layer
--- they touch different qubit blocks, so there is no collision --- and likewise
for the two halves of a band.

Depth is ``d_A + 2*ceil(d_B/2)``, which is ``Delta`` when ``d_B`` is even and
``Delta + 1`` when both are odd. For the gross code ``d_A = d_B = 3``, giving
depth 7 --- the value the SMT compiler of Zhang et al. (arXiv:2603.21499)
certifies as optimal for every BB code it reports.
"""
from __future__ import annotations

import numpy as np

from .bb import _poly
from .core import CSSCode, TannerGraph
from .schedule import Schedule

__all__ = ["bb_sandwich_schedule"]


def _monomial_index(l, m, terms, n_blk):
    """``idx[r, c]`` = which monomial of ``terms`` puts a 1 at ``(r, c)``, else -1."""
    idx = np.full((n_blk, n_blk), -1, dtype=int)
    for t, (i, j) in enumerate(terms):
        M = _poly(l, m, [(i, j)])
        rs, cs = np.nonzero(M)
        if (idx[rs, cs] != -1).any():
            raise ValueError("monomials overlap; exponents must be distinct")
        idx[rs, cs] = t
    return idx


def bb_sandwich_schedule(code: CSSCode, l: int, m: int, a_terms, b_terms,
                         merge: str = "A", perm_a=None, perm_b=None,
                         perm_a2=None, b_layout=None) -> Schedule:
    """The Eq. (S24) sandwich schedule for a BB code.

    ``merge`` picks which edge type is merged into the middle band; the SM shows
    the two choices give ``d_A + 2*ceil(d_B/2)`` and ``d_B + 2*ceil(d_A/2)``, so
    merging the *odd*-degree side is what reaches ``Delta``.

    ``perm_a`` and ``perm_b`` relabel which colour class occupies which slot:
    ``perm_a`` permutes the middle band's ``d_A`` layers, ``perm_b`` permutes
    the ``d_B`` early/late slots (so it also decides which colours go early).
    Every choice is a valid sandwich -- the band structure, and hence
    properness, is untouched -- but each gives a different CNOT order within
    each check, which is exactly the freedom Strikis, Browne and Beverland
    (arXiv:2603.05481) optimise over: the order fixes the residual (hook)
    errors and so the circuit distance.

    ``b_layout = (eb, lb, eb2, lb2)`` is the FULL monomial-coloured freedom of
    the split bands and overrides ``perm_b``: ``eb``/``lb`` list, layer by
    layer within the early/late band, which ``b`` colour occupies that layer
    (``None`` for none), and ``eb2``/``lb2`` do the same for the ``b*``
    colours. Properness needs ``b_j`` and ``b*_j`` in OPPOSITE
    bands (the L-side and R-side shared qubits of an overlapping check pair
    come in ``(a_i b_j, b_j a_i)`` pairs, and each pair must score 0 or 2), so
    ``set(eb2) == set(lb)`` and ``set(lb2) == set(eb)`` are asserted; within a
    band any placement is collision-free because each colour class is a
    permutation matrix on its own qubit block. Both split bands have width
    ``len(eb) == len(lb)``, so depth ``d_A + 2*ceil(d_B/2)`` is kept exactly
    when that width is ``ceil(d_B/2)``.
    """
    if merge not in ("A", "B"):
        raise ValueError("merge must be 'A' or 'B'")
    nblk = l * m
    ia = _monomial_index(l, m, a_terms, nblk)
    ib = _monomial_index(l, m, b_terms, nblk)
    dA, dB = len(a_terms), len(b_terms)
    if merge == "B":                      # relabel so the merged side is always 'A'
        ia, ib, dA, dB = ib, ia, dB, dA

    pa = list(range(dA)) if perm_a is None else list(perm_a)
    # `perm_a2` permutes the CONJUGATE (a*) layers independently of `a`. The
    # SM merges a and a* into one class, but properness only needs both in the
    # middle band: every overlapping pair then scores 1+1 or 0+0 whatever the
    # two layers are. Untying them multiplies the sandwich count by d_A!.
    pa2 = pa if perm_a2 is None else list(perm_a2)
    assert sorted(pa2) == list(range(dA))
    pb = list(range(dB)) if perm_b is None else list(perm_b)
    assert sorted(pa) == list(range(dA)) and sorted(pb) == list(range(dB))
    if b_layout is None:                  # Eq. (S24): pb-classes < ceil(dB/2) early
        e0 = (dB + 1) // 2
        inv = sorted(range(dB), key=lambda c: pb[c])      # colour with slot 0, 1, ...
        pad = lambda x: list(x) + [None] * (e0 - len(x))  # noqa: E731
        b_layout = (pad(inv[:e0]), pad(inv[e0:]), pad(inv[e0:]), pad(inv[:e0]))
    eb, lb, eb2, lb2 = (list(x) for x in b_layout)
    early = len(eb)                       # width of each split band
    assert len(lb) == len(eb2) == len(lb2) == early, "all four slot lists share the band width"
    cl = lambda x: [c for c in x if c is not None]  # noqa: E731
    assert sorted(cl(eb) + cl(lb)) == list(range(dB)) and sorted(cl(eb2) + cl(lb2)) == list(range(dB))
    assert set(cl(eb2)) == set(cl(lb)) and set(cl(lb2)) == set(cl(eb)), \
        "b_j and b*_j must sit in opposite bands"
    mid0, late0 = early, early + dA
    slot = {}                             # (colour, is_conjugate) -> layer
    for lst, conj, off in ((eb, False, 0), (eb2, True, 0), (lb, False, late0), (lb2, True, late0)):
        for k, c in enumerate(lst):
            if c is not None:
                slot[(c, conj)] = off + k
    graph = TannerGraph.build(code)
    times = np.full(len(graph.edges), -1, dtype=int)

    for e, (ck, q, ty) in enumerate(graph.edges):
        cid, left = ck[1], q < nblk
        qq = q if left else q - nblk
        # which family the edge belongs to, in Fig. S1's naming
        if merge == "A":
            a_edge = (ty == "X" and left) or (ty == "Z" and not left)
        else:
            a_edge = (ty == "X" and not left) or (ty == "Z" and left)
        if a_edge:                        # merged: middle band, colour = monomial
            col = (pa[ia[cid, qq]] if (ty == "X") else pa2[ia[qq, cid]])
            times[e] = mid0 + col
        else:                             # split: band and layer by (colour, conjugate)
            col = ib[cid, qq] if (ty == "X") else ib[qq, cid]
            times[e] = slot[(col, ty == "Z")]     # b* is the Z-side copy
    assert (times >= 0).all(), "some edge was never assigned a layer"
    T = int(times.max()) + 1
    return Schedule(code=code, graph=graph, times=times, T=T,
                    meta={"construction": f"sandwich(merge={merge})"})
