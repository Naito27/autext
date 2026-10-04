"""Shared core: CSSCode dimensions and the TannerGraph edge/overlap view."""
import numpy as np
import pytest

from autext import CSSCode, TannerGraph
from autext.core import _pairs_overlap


def _steane():
    H = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    return CSSCode(H, H, name="Steane")


def test_css_dimensions_and_delta():
    c = _steane()
    assert (c.n, c.rx, c.rz) == (7, 3, 3)
    assert c.is_css and c.k == 1
    # Delta = max(check weight, per-qubit X+Z count)
    assert c.delta == max(4, int((c.HX.sum(0) + c.HZ.sum(0)).max()))
    assert c.stats()["n"] == 7


def test_non_css_is_detected():
    HX = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    HZ = np.array([[1, 0, 1, 0]], dtype=np.uint8)      # overlap 1 -> not CSS
    assert not CSSCode(HX, HZ).is_css


def test_mismatched_widths_rejected():
    with pytest.raises(ValueError):
        CSSCode(np.zeros((1, 3), np.uint8), np.zeros((1, 4), np.uint8))


def test_tanner_graph_edges_match_matrices():
    c = _steane()
    g = TannerGraph.build(c)
    assert g.n_edges == int(c.HX.sum() + c.HZ.sum())
    for (ck, q, ty) in g.edges:
        H = c.HX if ty == "X" else c.HZ
        assert H[ck[1], q] == 1
    assert len(g.eidx) == g.n_edges


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_overlap_matches_bruteforce(seed):
    """The qubit-wise accumulation must agree with the O(r_X r_Z) definition."""
    rng = np.random.default_rng(seed)
    HX = (rng.random((6, 12)) < 0.3).astype(np.uint8)
    HZ = (rng.random((5, 12)) < 0.3).astype(np.uint8)
    brute = {}
    for a in range(HX.shape[0]):
        for b in range(HZ.shape[0]):
            shared = [int(q) for q in np.flatnonzero(HX[a] & HZ[b])]
            if shared:
                brute[(a, b)] = shared
    assert _pairs_overlap(HX, HZ) == brute
