"""The two reference schedulers of Eq. (3): staggered upper bound and CP-SAT optimum."""
import numpy as np
import pytest

from autext import CSSCode, exact_schedule, staggered_schedule


def _steane():
    H = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    return CSSCode(H, H, name="Steane")


def test_staggered_is_proper_at_deltaX_plus_deltaZ():
    c = _steane()
    s = staggered_schedule(c)
    DX, DZ = s.meta["DX"], s.meta["DZ"]
    assert s.T == DX + DZ
    assert s.verify_proper() and s.verify_contamination()
    # all X gates strictly precede all Z gates
    tx = [s.times[e] for e, (_c, _q, ty) in enumerate(s.graph.edges) if ty == "X"]
    tz = [s.times[e] for e, (_c, _q, ty) in enumerate(s.graph.edges) if ty == "Z"]
    assert max(tx) < min(tz)


def test_staggered_beats_or_meets_the_upper_bound_of_eq3():
    c = _steane()
    s = staggered_schedule(c)
    assert c.delta <= s.depth <= s.meta["DX"] + s.meta["DZ"]


def test_exact_schedule_is_valid_and_at_least_delta():
    pytest.importorskip("ortools")
    c = _steane()
    s = exact_schedule(c, time_limit=20.0)
    assert s is not None
    assert s.is_valid()
    assert s.depth >= c.delta
    if s.meta["optimal"]:
        assert s.depth <= staggered_schedule(c).depth
