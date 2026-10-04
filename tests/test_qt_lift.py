"""The base -> lifted bridge: does a verified base schedule lift to a proper,
depth-optimal circuit on the actual quantum Tanner code?

This is the end-to-end check of the paper's claim for QT codes: the reduced
problem is solved on ``T^(base)_QT``, and inverting ``P_lift`` must give a valid
parity-check circuit on the full lifted code of Eq. (S48) at the *same* depth.
"""
import numpy as np
import pytest

from autext.qt import qt
from autext.qt.group import Group
from autext.qt.lift import base_edge_keys, lift_and_verify, lifted_code
from test_qt_base import find_base_schedule


GROUPS = [Group.cyclic(2), Group.cyclic(3), Group.dihedral(3)]


def test_group_tables_are_sane():
    for G in GROUPS + [Group.cyclic(5), Group.dihedral(4)]:
        L = len(G)
        assert G.mult.shape == (L, L)
        # associativity on a sample, and inverses
        for a in range(L):
            assert G.mult[a, G.inv[a]] == 0 and G.mult[G.inv[a], a] == 0
            for b in range(L):
                assert G.mult[G.mult[a, b], 0] == G.mult[a, b]
        # every row/column is a permutation (Latin square)
        assert all(len(set(G.mult[a])) == L for a in range(L))
        assert all(len(set(G.mult[:, b])) == L for b in range(L))


def test_dihedral_is_non_abelian():
    G = Group.dihedral(3)
    assert any(G.mult[a, b] != G.mult[b, a] for a in range(len(G)) for b in range(len(G)))


@pytest.mark.parametrize("name", ["[4,2,2]", "[6,3,3]"])
@pytest.mark.parametrize("gi", range(len(GROUPS)))
def test_lifted_code_is_css_and_schedule_lifts(name, gi):
    G = GROUPS[gi]
    Q = qt.QT(name, name)
    got = find_base_schedule(Q)
    assert got is not None
    inst, sch, T = got

    rng = np.random.default_rng(gi)
    A = rng.integers(0, len(G), Q.nA)
    B = rng.integers(0, len(G), Q.nB)
    out = lift_and_verify(Q.pair, Q.nA, Q.nB, G, A, B, sch, T)

    code, rep = out["code"], out["report"]
    # commutation is automatic by construction (SM S51) -- for any group/multiset
    assert out["is_css"]
    assert code.n == Q.nA * Q.nB * len(G)
    # the lift preserves the depth of the reduced problem
    assert out["depth"] == T == code.delta
    # proper by the Eq. (2) parity formula AND by independent X-propagation
    assert rep["proper"] and rep["contam"] and rep["sim"]
    assert rep["sim_bad"] == []


def test_base_edge_keys_align_with_build_order():
    """The key list must index-align with the instance the solver schedules."""
    from autext.qt import build

    Q = qt.QT("[4,2,2]", "[6,3,3]")
    rho = qt.weight_class_halving(Q.H)
    sigma = qt.weight_class_halving(Q.Hp)
    inst = build.from_qt(Q, rho, sigma)
    keys = base_edge_keys(Q.pair, Q.nA, Q.nB)
    assert len(keys) == inst.m
    # the square recorded by build must be the (i, j) in the key
    for e in range(0, inst.m, max(1, inst.m // 50)):
        _f, _rA, _rB, i, j = keys[e]
        assert int(inst.sq[e]) == i * Q.nB + j


def test_trivial_group_reproduces_the_base_code():
    """|G| = 1 collapses the lift: the code is the base CSS code of Eq. (S45)."""
    Q = qt.QT("[4,2,2]", "[4,2,2]")
    G = Group.cyclic(1)
    code, _meta = lifted_code(Q.pair, Q.nA, Q.nB, G, [0] * Q.nA, [0] * Q.nB)
    assert code.is_css and code.n == Q.nA * Q.nB
    # the X0 block is exactly H0 (x) G'0 of the base code, Eq. (S45)
    blk = (np.kron(Q.pair["X0"][0], Q.pair["X0"][1]) % 2).astype(np.uint8)
    assert np.array_equal(code.HX[: blk.shape[0]], blk)


def test_lift_rejects_wrong_length_schedule():
    Q = qt.QT("[4,2,2]", "[4,2,2]")
    G = Group.cyclic(2)
    with pytest.raises(ValueError):
        lift_and_verify(Q.pair, Q.nA, Q.nB, G, [0] * Q.nA, [0] * Q.nB,
                        np.ones(3, int), 4)


def test_coupled_pi_still_gives_a_css_code_that_lifts():
    """C_0 = C_1 with a non-trivial relabelling is a genuine QT code: the
    coupled (pi_A, pi_B) keeps every block of HX HZ^T zero, and a depth-Delta
    base schedule still lifts to a proper depth-Delta circuit."""
    G = GROUPS[0]
    rng = np.random.default_rng(4)
    Q = qt.QT.random("[7,4,3]", "[5,2,3]", rng)
    assert not Q.is_A
    got = find_base_schedule(Q)
    assert got is not None
    inst, sch, T = got
    A = rng.integers(0, len(G), Q.nA)
    B = rng.integers(0, len(G), Q.nB)
    out = lift_and_verify(Q.pair, Q.nA, Q.nB, G, A, B, sch, T)
    assert out["is_css"]
    assert out["depth"] == T
    rep = out["report"]
    assert rep["proper"] and rep["contam"] and rep["sim"]


def test_four_independent_permutations_are_not_a_qt_code():
    """Permuting the four families independently is a strict superset of the
    construction, not a generalisation of it: commutation ties H_1's
    relabelling to G_1's, so independent permutations leave HX HZ^T != 0 and
    the lifted code is not CSS."""
    from autext.qt import codes
    from autext.qt.lift import lifted_code
    from autext.qt.group import Group
    rng = np.random.default_rng(0)
    G, H = codes.get_code("[6,3,3]")
    Gp, Hp = codes.get_code("[6,3,3]")
    f = lambda M: M[:, rng.permutation(M.shape[1])]
    bad = {"X0": (f(H), f(Gp)), "X1": (f(H), f(Gp)),
           "Z0": (f(G), f(Hp)), "Z1": (f(G), f(Hp))}
    code, _ = lifted_code(bad, 6, 6, Group.cyclic(1), [0] * 6, [0] * 6)
    assert not code.is_css


def test_group_from_perms_matches_the_named_constructors():
    """``Group.from_perms`` is public API; pin it against cyclic/dihedral."""
    from autext.qt.group import Group
    # the 3-cycle generates C3
    c3 = Group.from_perms([(1, 2, 0)])
    assert len(c3) == 3
    # S3 = D3: a 3-cycle and a transposition
    d3 = Group.from_perms([(1, 2, 0), (1, 0, 2)])
    assert len(d3) == 6
    for G in (c3, d3):
        tbl = np.asarray(G.mult)
        for r in tbl:                       # Latin square
            assert sorted(r.tolist()) == list(range(len(G)))
        for a in range(len(G)):
            assert G.mult[a, G.inv[a]] == 0 and G.mult[G.inv[a], a] == 0
    # C3 is abelian, D3 is not
    assert np.array_equal(c3.mult, c3.mult.T)
    assert not np.array_equal(d3.mult, d3.mult.T)
    # and it lifts like any other group
    Q = qt.QT("[4,2,2]", "[4,2,2]")
    got = find_base_schedule(Q)
    assert got is not None
    _inst, sch, T = got
    out = lift_and_verify(Q.pair, Q.nA, Q.nB, d3, [0] * Q.nA, [0] * Q.nB, sch, T)
    assert out["is_css"] and out["depth"] == T


def test_simulate_checks_both_propagation_directions():
    """The Z pass is redundant on a CSS code (Zleak == Xleak, since commutation
    makes every overlap even) and is kept only as defence in depth.  This pins
    the behaviour either way: it must agree with the X pass wherever the
    equivalence applies, and must not silently pass a non-CSS code."""
    from autext.core import CSSCode, TannerGraph
    from autext.validation import simulate_contamination

    rng = np.random.default_rng(0)
    seen_noncss = 0
    for _ in range(200):
        HX = (rng.random((3, 6)) < 0.5).astype(np.uint8)
        HZ = (rng.random((3, 6)) < 0.5).astype(np.uint8)
        code = CSSCode(HX, HZ)
        graph = TannerGraph.build(code)
        if graph.n_edges == 0:
            continue
        times = rng.integers(0, 8, graph.n_edges)
        ok, bad = simulate_contamination(code, times, graph)
        assert ok == (len(bad) == 0)
        if not code.is_css:
            seen_noncss += 1
            assert not ok            # the Z pass always fires here
    assert seen_noncss > 0

    # and on a real CSS QT code a clean schedule stays clean under both passes
    Q = qt.QT("[6,3,3]", "[6,3,3]")
    got = find_base_schedule(Q)
    assert got is not None
    _inst, sch, T = got
    out = lift_and_verify(Q.pair, Q.nA, Q.nB, GROUPS[0],
                          [0] * Q.nA, [0] * Q.nB, sch, T)
    assert out["is_css"] and out["report"]["sim"] and not out["report"]["sim_bad"]
