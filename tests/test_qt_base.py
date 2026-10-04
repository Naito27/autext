"""The QT base graph: quadrant sandwich, instance invariants, and the verifier.

Everything here is on ``T^(base)_QT`` -- the quotient left by inverting the
group lift.
"""
import numpy as np
import pytest

from autext.qt import build, core, qt, simple


def find_base_schedule(Q, tries=8, seeds=6):
    """A halving + seed reaching depth Delta, or None. Returns (inst, sch, T)."""
    D = Q.Delta()
    for rho in qt.balanced_halvings(Q.nA)[:tries]:
        for sigma in qt.balanced_halvings(Q.nB)[:tries]:
            inst = build.from_qt(Q, rho, sigma)
            if not core.straddles(inst, D):
                continue                        # depth Delta provably impossible
            for s in range(seeds):
                sch = simple.run(inst, D, seed=s)
                if sch is not None:
                    return inst, sch, D
    return None


# -- the sandwich ---------------------------------------------------------
def test_every_quadrant_is_a_sandwich():
    seen_early = set()
    for q in range(4):
        E, M, L = qt.roles(q)
        assert len({E, *M, L}) == 4          # one early, two middle, one late
        assert set([E, *M, L]) == set(qt.FAMS)
        seen_early.add(E)
        assert qt.roles(3 - q)[2] == E       # late in the antipodal quadrant
    assert seen_early == set(qt.FAMS)        # each family early exactly once


def test_precedences_are_acyclic_for_all_four_halvings():
    for a in (0, 1):
        for b in (0, 1):
            prec = qt.precedences(a, b)
            before = {f: 0 for f in qt.FAMS}
            for _f1, f2 in prec:
                before[f2] += 1
            assert sorted(before.values()) == [0, 1, 1, 2]


# -- instance invariants --------------------------------------------------
@pytest.mark.parametrize("name", ["[4,2,2]", "[6,3,3]"])
def test_instance_matches_closed_form_delta(name):
    Q = qt.QT(name, name)
    rho = qt.weight_class_halving(Q.H)
    sigma = qt.weight_class_halving(Q.Hp)
    inst = build.from_qt(Q, rho, sigma)
    assert inst.nsq == Q.nA * Q.nB
    assert inst.Delta() == Q.Delta()             # graph degree == closed form
    assert inst.m == sum(int(MA.sum()) * int(MB.sum()) for MA, MB in Q.pair.values())
    assert set(np.unique(inst.grp)) <= {1, 2, 3}


# -- the referee ----------------------------------------------------------
def test_layer_sweep_reaches_delta_and_verifies():
    Q = qt.QT("[6,3,3]", "[6,3,3]")
    got = find_base_schedule(Q)
    assert got is not None, "no depth-Delta schedule found in the swept halvings"
    inst, sch, T = got
    assert core.verify(inst, sch, T)
    assert sch.min() >= 1 and sch.max() <= T     # layers are 1-based


def test_verify_rejects_a_collision():
    Q = qt.QT("[4,2,2]", "[4,2,2]")
    got = find_base_schedule(Q)
    assert got is not None
    inst, sch, T = got
    bad = sch.copy()
    # force two edges at the same square into one layer
    order = np.lexsort((bad, inst.sq))
    for a, b in zip(order, order[1:]):
        if inst.sq[a] == inst.sq[b]:
            bad[b] = bad[a]
            break
    with pytest.raises(AssertionError):
        core.verify(inst, bad, T)


def test_straddles_is_necessary_for_depth_delta():
    """Any halving admitting a depth-Delta schedule must pass the straddling test."""
    Q = qt.QT("[4,2,2]", "[4,2,2]")
    D = Q.Delta()
    checked = 0
    for rho in qt.balanced_halvings(Q.nA)[:6]:
        for sigma in qt.balanced_halvings(Q.nB)[:6]:
            inst = build.from_qt(Q, rho, sigma)
            for s in range(4):
                sch = simple.run(inst, D, seed=s)
                if sch is not None:
                    core.verify(inst, sch, D)
                    assert core.straddles(inst, D)
                    checked += 1
                    break
    assert checked > 0


# --------------------------------------------------------------------------
# the coupled relabelling (pi_A, pi_B): assumption [A] is pi = id, and is NOT
# what "C_0 = C_1" means -- the two copies carry independent labellings
# --------------------------------------------------------------------------

def test_default_is_assumption_A_and_matches_the_old_construction():
    from autext.qt import codes
    for a in ("[4,2,2]", "[6,3,3]", "[8,4,4]"):
        for b in ("[2,1,2]", "[7,4,3]"):
            Q = qt.QT(a, b)
            G, H = codes.get_code(a)
            Gp, Hp = codes.get_code(b)
            assert Q.is_A
            old = {"X0": (H, Gp), "X1": (H, Gp), "Z0": (G, Hp), "Z1": (G, Hp)}
            for f in old:
                for i in (0, 1):
                    assert np.array_equal(Q.pair[f][i], old[f][i])


def test_random_pi_breaks_A_and_splits_the_family_degrees():
    rng = np.random.default_rng(0)
    Q = qt.QT.random("[6,3,3]", "[6,3,3]", rng)
    assert not Q.is_A
    # under [A] deg_X0 == deg_X1 and deg_Z0 == deg_Z1 everywhere; a non-trivial
    # relabelling is exactly what breaks that
    assert np.array_equal(qt.QT("[6,3,3]", "[6,3,3]").deg_f("X0"),
                          qt.QT("[6,3,3]", "[6,3,3]").deg_f("X1"))
    assert not np.array_equal(Q.deg_f("X0"), Q.deg_f("X1"))


def test_pi_must_be_a_permutation():
    with pytest.raises(AssertionError):
        qt.QT("[4,2,2]", "[4,2,2]", piA=[0, 1, 2])          # wrong length
    with pytest.raises(AssertionError):
        qt.QT("[4,2,2]", "[4,2,2]", piA=[0, 0, 1, 2])       # not a bijection


def test_bands_closed_form_refuses_non_identity_pi():
    from autext.qt.bandopt import Bands
    Bands(qt.QT("[6,3,3]", "[6,3,3]"))                      # [A]: fine
    rng = np.random.default_rng(1)
    with pytest.raises(AssertionError):
        Bands(qt.QT.random("[6,3,3]", "[6,3,3]", rng))


# --------------------------------------------------------------------------
# every builder must produce a real CSS code: commutation ties the relabelling
# of H_1 to that of G_1, so the four families cannot be permuted independently
# --------------------------------------------------------------------------

def test_commutes_agrees_with_the_kronecker_computation():
    """``qt.commutes`` checks the small factors; pin it against HX HZ^T = 0."""
    from autext.qt import codes

    def by_kron(p):
        HX = np.vstack([np.kron(*p["X0"]), np.kron(*p["X1"])])
        HZ = np.vstack([np.kron(*p["Z0"]), np.kron(*p["Z1"])])
        return not (HX.dot(HZ.T) % 2).any()

    rng = np.random.default_rng(0)
    for a in ("[4,2,2]", "[6,3,3]", "[7,4,3]"):
        for b in ("[4,2,2]", "[5,2,3]"):
            for Q in (qt.QT(a, b), qt.QT.random(a, b, rng)):
                assert qt.commutes(Q.pair) is True
                assert by_kron(Q.pair) is True
            # independently permuted families: both must agree it is broken
            G, H = codes.get_code(a)
            Gp, Hp = codes.get_code(b)
            f = lambda M: M[:, rng.permutation(M.shape[1])]
            bad = {"X0": (f(H), f(Gp)), "X1": (f(H), f(Gp)),
                   "Z0": (f(G), f(Hp)), "Z1": (f(G), f(Hp))}
            # exact, so it must track the Kronecker computation even when a
            # broken pairing commutes by accident through the other factor
            assert qt.commutes(bad) == by_kron(bad)


def test_pairs_from_codes_drops_A_and_stays_css():
    """Two independent codes per side: [A] gone, commutation intact."""
    from autext.qt.random_codes import random_code
    import random as _random
    pyr = _random.Random(0)
    for _ in range(5):
        A0, A1 = random_code(pyr, 8, 4), random_code(pyr, 8, 4)
        B0, B1 = random_code(pyr, 8, 4), random_code(pyr, 8, 4)
        if None in (A0, A1, B0, B1):
            continue
        p = qt.pairs_from_codes(A0, A1, B0, B1)
        assert qt.commutes(p)
        assert not np.array_equal(p["X0"][0], p["X1"][0])      # H0 != H1


def test_every_stress_builder_makes_a_css_code():
    """Each driver's pairs builder, exercised once at a small size."""
    from autext.qt import stress
    assert qt.commutes(stress.code_pairs(8, 4, 8, 4, 0))
    assert qt.commutes(stress.code_pairs(8, 4, 8, 4, 0, degenerate=True))
    ident = stress.code_pairs(8, 4, 8, 4, 0, identical=True)
    assert qt.commutes(ident)
    assert np.array_equal(ident["X0"][0], ident["X1"][0])       # the [A] control
    ep = stress.expander_pairs(12, 3, 6, 12, 3, 6, 0)
    if ep is not None:
        assert qt.commutes(ep)


def test_ref36_convention_and_mixed_local_codes():
    """Leverrier, Rozendaal and Zemor (arXiv:2512.20532) specify C_i = ker H_i
    with the rows of G_i spanning C_i, and take all four local codes
    independently."""
    from autext.qt import codes

    for name in list(codes.P_OF) + list(codes.EXTRA_P):
        G, H = codes.get_code(name)
        n, k, d = (int(x) for x in name.strip("[]").split(","))
        assert G.shape == (k, n) and H.shape == (n - k, n)
        assert not (H.dot(G.T) % 2).any()          # H_i G_i^T = 0
        assert codes.min_distance(G) == d          # the name is not a claim

    # four genuinely different codes, and still a CSS code
    Q = qt.QT.mixed("[6,3,3]", "[6,2,4]", "[6,4,2]", "[6,3,3]")
    assert qt.commutes(Q.pair)
    assert not Q.is_A
    assert not np.array_equal(Q.pair["X0"][0], Q.pair["X1"][0])     # H0 != H1
    assert not np.array_equal(Q.pair["X0"][1], Q.pair["X1"][1])     # G'0 != G'1
    # the default two-name form is unchanged and still satisfies [A]
    assert qt.QT("[6,3,3]", "[6,3,3]").is_A
    # a side cannot mix two different lengths
    with pytest.raises(ValueError):
        qt.QT.mixed("[6,3,3]", "[7,4,3]", "[6,3,3]", "[6,3,3]")


def test_ref36_qldpc_criterion():
    """Leverrier et al. select local codes whose dual has IDENTICAL parameters,
    then discard generator weight 4 as toric-like.  Both conditions are
    structural, so pin them and the resulting pair list."""
    from autext.qt import codes, ldpc

    # condition 1 picks out exactly the four codes -- the three names used by
    # Leverrier et al. ([2,1,2], [6,3,3], [8,4,4]) plus [4,2,2]
    assert ldpc.eligible_codes() == ["[2,1,2]", "[4,2,2]", "[6,3,3]", "[8,4,4]"]
    for nm in ldpc.eligible_codes():
        G, H = codes.get_code(nm)
        assert H.shape[0] == G.shape[0]                       # k = n-k
        assert codes.min_distance(G) == codes.min_distance(H)  # d = d^perp
    # a code with a dense dual must not qualify: it makes the Z-families dense
    assert not ldpc.self_dual_parameters("[7,4,3]")

    # the paper's own count of distinct column-permuted codes
    assert len(ldpc.perm_classes("[6,3,3]")) == 30
    assert len(ldpc.perm_classes("[8,4,4]")) == 30
    assert len(ldpc.perm_classes("[2,1,2]")) == 1

    # weights: the three combinations of Leverrier et al. are 6, 8 and 9
    assert ldpc.generator_weight("[6,3,3]", "[2,1,2]") == 6
    assert ldpc.generator_weight("[8,4,4]", "[2,1,2]") == 8
    assert ldpc.generator_weight("[6,3,3]", "[6,3,3]") == 9
    # and the weight-4 corner is excluded
    assert ldpc.generator_weight("[2,1,2]", "[2,1,2]") == 4
    assert all(w >= 6 for _a, _b, w in ldpc.qldpc_pairs())
    for pair in ldpc.REF36_PAIRS:
        assert any((a, b) == pair for a, b, _w in ldpc.qldpc_pairs())
