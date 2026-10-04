"""Build the lifted quantum Tanner code, and lift a base schedule onto it.

Everything else in :mod:`autext.qt` works on the **base** graph
``T^(base)_QT`` of Eq. (S45) -- the quotient left behind when the group lift is
inverted by the edge partition ``P_lift``. This module closes the loop back to
the *actual* code:

1. :func:`lifted_code` assembles the QT parity checks of Eq. (S48) from the four
   family matrix pairs, a group ``G`` and the multisets ``A``, ``B``;
2. :func:`lift_times` inverts ``P_lift`` (Eq. S55) -- every lifted edge inherits
   the CNOT layer of its base edge, since a base edge class *is* the set of its
   ``|G|`` lifted copies;
3. :func:`lift_and_verify` runs the resulting circuit through the full-code
   verifier (properness by the Eq. (2) parity formula, plus the independent
   X-propagation simulation).

The family matrix pairs follow Eq. (S45)/(S48)::

    X0 = (H0, G'0)    X1 = (H1, G'1)    Z0 = (G0, H'1)    Z1 = (G1, H'0)

which is exactly the layout of :attr:`autext.qt.qt.QT.pair`. Per Eq. (S49) the
qubit reached by a family-``f`` check at group coordinate ``g`` on square
``(i, j)`` is ``Q(i, j, phi_f(g, i, j))`` with

    phi_X0 = g          phi_Z1 = a_i^-1 g
    phi_Z0 = g b_j      phi_X1 = a_i^-1 g b_j

Indexing: qubit ``(i, j, g)`` is ``(i*nB + j)*|G| + g``; a family block's check
``(rA, rB, g)`` is ``off_f + (rA*nrB + rB)*|G| + g``, with X blocks numbered
inside ``H_X`` and Z blocks inside ``H_Z``.
"""
from __future__ import annotations

import numpy as np

from ..core import CSSCode, TannerGraph
from ..schedule import Schedule
from .qt import FAMS

__all__ = ["lifted_code", "base_edge_keys", "lift_times", "lift_and_verify"]

XFAMS, ZFAMS = ("X0", "X1"), ("Z0", "Z1")


def _phi(fam, g, i, j, mult, inv, A, B):
    """Group coordinate of the qubit reached on square ``(i, j)`` -- Eq. (S49).

    ``g`` may be an array (the whole group coordinate at once).
    """
    if fam == "X0":
        return g
    if fam == "Z1":
        return mult[inv[A[i]], g]
    if fam == "Z0":
        return mult[g, B[j]]
    if fam == "X1":
        return mult[mult[inv[A[i]], g], B[j]]
    raise KeyError(fam)


def _blocks(pairs, fams, L):
    """``[(fam, nrA, nrB, offset)]`` and the total row count for one side."""
    out, off = [], 0
    for f in fams:
        MA, MB = pairs[f]
        nrA, nrB = int(MA.shape[0]), int(MB.shape[0])
        out.append((f, nrA, nrB, off))
        off += nrA * nrB * L
    return out, off


def lifted_code(pairs, nA, nB, group, A, B, name: str = "") -> tuple[CSSCode, dict]:
    """Assemble the lifted QT code of Eq. (S48).

    ``pairs`` maps each family to its ``(MA, MB)``; ``A``/``B`` are multisets of
    group-element indices of length ``nA``/``nB``. Returns ``(code, meta)``,
    where ``meta`` carries the block layout needed to decode check indices.
    """
    L = len(group)
    mult, inv = group.mult, group.inv
    A = np.asarray(A, dtype=int)
    B = np.asarray(B, dtype=int)
    if A.shape != (nA,) or B.shape != (nB,):
        raise ValueError(f"A must have length {nA} and B length {nB}")

    nq = nA * nB * L
    xb, nX = _blocks(pairs, XFAMS, L)
    zb, nZ = _blocks(pairs, ZFAMS, L)
    HX = np.zeros((nX, nq), dtype=np.uint8)
    HZ = np.zeros((nZ, nq), dtype=np.uint8)
    gs = np.arange(L)

    for H, blocks in ((HX, xb), (HZ, zb)):
        for fam, nrA, nrB, off in blocks:
            MA, MB = pairs[fam]
            rAs, iAs = np.nonzero(MA)
            rBs, jBs = np.nonzero(MB)
            for rA, i in zip(rAs, iAs):
                for rB, j in zip(rBs, jBs):
                    chk = off + (int(rA) * nrB + int(rB)) * L + gs
                    qub = (int(i) * nB + int(j)) * L + _phi(fam, gs, int(i), int(j),
                                                            mult, inv, A, B)
                    H[chk, qub] ^= 1

    code = CSSCode(HX, HZ, name=name or f"QT(lift |G|={L})")
    meta = dict(nA=nA, nB=nB, L=L, xblocks=xb, zblocks=zb,
                group=group, A=A, B=B, pairs=pairs)
    return code, meta


def base_edge_keys(pairs, nA, nB) -> list[tuple]:
    """``(fam, rA, rB, i, j)`` per base edge, in :func:`autext.qt.build.build` order.

    ``build`` emits families in ``FAMS`` order and, within a family, iterates the
    nonzeros of ``MA`` in the outer loop and of ``MB`` in the inner one; this
    mirrors that exactly.  Contract: this list must stay index-aligned with the
    edge order ``build`` produces, since base schedules are indexed by it -- a
    change to one edge order requires the same change here.
    """
    keys: list[tuple] = []
    for f in FAMS:
        MA, MB = pairs[f]
        rAs, iAs = np.nonzero(MA)
        rBs, jBs = np.nonzero(MB)
        for rA, i in zip(rAs, iAs):
            for rB, j in zip(rBs, jBs):
                keys.append((f, int(rA), int(rB), int(i), int(j)))
    return keys


def _decode_check(meta, ctype, cid):
    blocks = meta["xblocks"] if ctype == "X" else meta["zblocks"]
    L = meta["L"]
    for fam, nrA, nrB, off in blocks:
        if off <= cid < off + nrA * nrB * L:
            loc = cid - off
            rest = loc // L
            return fam, rest // nrB, rest % nrB
    raise KeyError((ctype, cid))


def lift_times(code, meta, base_sched, pairs=None):
    """Transport a base schedule onto the lifted code (inverse of ``P_lift``).

    ``base_sched`` is the 1-based per-base-edge layer array the QT solvers
    return; the result is 0-based layers aligned with ``TannerGraph.build(code).edges``.
    """
    pairs = meta["pairs"] if pairs is None else pairs
    nA, nB, L = meta["nA"], meta["nB"], meta["L"]
    keys = base_edge_keys(pairs, nA, nB)
    base_sched = np.asarray(base_sched)
    if len(keys) != len(base_sched):
        raise ValueError(f"schedule has {len(base_sched)} edges, base graph has {len(keys)}")
    tof = {k: int(t) for k, t in zip(keys, base_sched)}

    graph = TannerGraph.build(code)
    times = np.empty(graph.n_edges, dtype=int)
    for e, (ck, q, _ty) in enumerate(graph.edges):
        ctype, cid = ck
        fam, rA, rB = _decode_check(meta, ctype, cid)
        s = q // L
        times[e] = tof[(fam, rA, rB, s // nB, s % nB)] - 1      # 1-based -> 0-based
    return graph, times


def lift_and_verify(pairs, nA, nB, group, A, B, base_sched, T, name: str = "") -> dict:
    """Build the lifted code, lift ``base_sched`` onto it, and verify the circuit.

    Returns a dict with the code, the :class:`~autext.schedule.Schedule`, and the
    verifier's report (``proper`` / ``contam`` / ``sim``).
    """
    code, meta = lifted_code(pairs, nA, nB, group, A, B, name=name)
    graph, times = lift_times(code, meta, base_sched, pairs)
    sched = Schedule(code=code, graph=graph, times=times, T=T,
                     meta=dict(method="qt-lift", T=T, group=repr(group)))
    return dict(code=code, meta=meta, schedule=sched,
                is_css=code.is_css, depth=sched.depth, report=sched.verify())
