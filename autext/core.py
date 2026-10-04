"""Core data model: CSS codes and their Tanner graphs.

:class:`CSSCode` is a pair of parity-check matrices; :class:`TannerGraph` is the
derived edge/overlap view that the schedule verifier consumes. An *edge* is one
CX gate, ``(check, qubit, check-type)``, and its position in ``edges`` is its
stable integer id.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linalg import rank_gf2

__all__ = ["CSSCode", "Edge", "TannerGraph"]


@dataclass
class CSSCode:
    """A CSS code given by parity-check matrices ``H_X`` and ``H_Z``."""

    HX: np.ndarray
    HZ: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.HX = (np.asarray(self.HX) % 2).astype(np.uint8)
        self.HZ = (np.asarray(self.HZ) % 2).astype(np.uint8)
        if self.HX.shape[1] != self.HZ.shape[1]:
            raise ValueError("H_X and H_Z must have the same number of columns (qubits)")

    # -- basic dimensions ------------------------------------------------
    @property
    def n(self) -> int:
        """Number of physical (data) qubits."""
        return self.HX.shape[1]

    @property
    def rx(self) -> int:
        """Number of X-checks (rows of ``H_X``)."""
        return self.HX.shape[0]

    @property
    def rz(self) -> int:
        """Number of Z-checks (rows of ``H_Z``)."""
        return self.HZ.shape[0]

    @property
    def k(self) -> int:
        """Number of logical qubits, ``n - rank H_X - rank H_Z``."""
        return self.n - rank_gf2(self.HX) - rank_gf2(self.HZ)

    @property
    def delta(self) -> int:
        """Max Tanner-graph degree ``Delta`` -- the Koenig floor on CNOT depth.

        The maximum over X-check weights, Z-check weights, and the total CX
        count per data qubit (X plus Z).
        """
        return int(
            max(
                self.HX.sum(1).max(initial=0),
                self.HZ.sum(1).max(initial=0),
                (self.HX.sum(0) + self.HZ.sum(0)).max(initial=0),
            )
        )

    @property
    def is_css(self) -> bool:
        """True iff ``H_X H_Z^T = 0 (mod 2)``."""
        return int((self.HX.astype(np.int64) @ self.HZ.T.astype(np.int64) % 2).sum()) == 0

    def stats(self) -> dict:
        """Summary dictionary (qubits, checks, weights, ``Delta``, ``k``)."""
        return dict(
            name=self.name, n=self.n, rx=self.rx, rz=self.rz,
            max_xw=int(self.HX.sum(1).max(initial=0)),
            max_zw=int(self.HZ.sum(1).max(initial=0)),
            max_qubit_deg=int((self.HX.sum(0) + self.HZ.sum(0)).max(initial=0)),
            delta=self.delta, k=self.k, is_css=self.is_css,
        )

    def __repr__(self) -> str:
        tag = self.name or "CSSCode"
        return f"{tag}(n={self.n}, rX={self.rx}, rZ={self.rz}, Delta={self.delta})"


# --------------------------------------------------------------------------
# Tanner graph
# --------------------------------------------------------------------------
# An edge / CX gate: (check, qubit, check-type). ``check`` is (ctype, cid).
# X-edges come first, then Z-edges, so an edge's position in ``edges`` is its
# stable integer id.
Edge = tuple


@dataclass
class TannerGraph:
    """Edges, edge lookup, and X/Z check overlaps for a :class:`CSSCode`."""

    code: CSSCode
    edges: list[Edge]
    eidx: dict[tuple, int]
    overlaps: dict[tuple[int, int], list[int]]

    @classmethod
    def build(cls, code: CSSCode) -> "TannerGraph":
        edges = _build_edges(code.HX, code.HZ)
        eidx = {(ck, q): i for i, (ck, q, _ty) in enumerate(edges)}
        overlaps = _pairs_overlap(code.HX, code.HZ)
        return cls(code=code, edges=edges, eidx=eidx, overlaps=overlaps)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def _build_edges(HX: np.ndarray, HZ: np.ndarray) -> list[Edge]:
    edges: list[Edge] = []
    for a in range(HX.shape[0]):
        for q in np.nonzero(HX[a])[0]:
            edges.append((("X", a), int(q), "X"))
    for b in range(HZ.shape[0]):
        for q in np.nonzero(HZ[b])[0]:
            edges.append((("Z", b), int(q), "Z"))
    return edges


def _pairs_overlap(HX: np.ndarray, HZ: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each overlapping ``(X-check a, Z-check b)`` to its shared qubits.

    Accumulated qubit-by-qubit, so the cost is ``sum_q deg_X(q) deg_Z(q)`` --
    linear in the number of overlapping incidences rather than ``O(r_X r_Z)``.
    Shared-qubit lists come out ascending because ``q`` is scanned in order.
    """
    overlaps: dict[tuple[int, int], list[int]] = {}
    for q in range(HX.shape[1]):
        xs = np.nonzero(HX[:, q])[0]
        if xs.size == 0:
            continue
        zs = np.nonzero(HZ[:, q])[0]
        for a in xs:
            for b in zs:
                overlaps.setdefault((int(a), int(b)), []).append(q)
    return overlaps
