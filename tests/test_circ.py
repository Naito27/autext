"""autext.circ: pinned against stim's oracle, and the bound directions enforced."""
import numpy as np
import pytest

stim = pytest.importorskip("stim")

from autext.bb import CATALOGUE, bb_code, from_catalogue
from autext.circ import (Bound, bracket, deinterleave, dem, dem_matrices, drop_dead,
                         exact_cpsat, lower_ext, memory_circuit, min_over_bases,
                         upper_stim)
from autext.circ.bounds import Bracket
from autext.lp_sandwich import bb_sandwich_schedule
from autext.schedule import Schedule


def _surface(d):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=1,
        after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3,
        after_reset_flip_probability=1e-3, before_round_data_depolarization=1e-3)


def _bb72():
    code = from_catalogue("[[72,12,6]]")
    s = bb_sandwich_schedule(code, *CATALOGUE["[[72,12,6]]"])
    return code, s


# ----------------------------------------------------------------- oracle
@pytest.mark.parametrize("d", [3, 5])
def test_cpsat_matches_stim_oracle(d):
    circ = _surface(d)
    D, O = drop_dead(*dem_matrices(circ.detector_error_model(decompose_errors=False)))
    ours = exact_cpsat(D, O, time_limit=120, ub=d)
    theirs = len(circ.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False))
    assert ours.kind == "exact" and ours.value == theirs == d


# ------------------------------------------------------------ bound types
def test_bound_kinds_and_bracket_directions():
    up = Bound(6, "upper", basis="Z")
    lo = Bound(3, "lower", basis="Z")
    assert up.hi == 6 and up.lo is None and lo.lo == 3 and lo.hi is None
    br = bracket(up, lo, basis="Z")
    assert (br.lo, br.hi, br.exact) == (3, 6, None)
    assert bracket(Bound(6, "lower", basis="Z"), up, basis="Z").exact == 6
    with pytest.raises(AssertionError):            # lower above upper = solver bug
        bracket(Bound(7, "lower", basis="Z"), up, basis="Z")
    with pytest.raises(ValueError):
        Bound(1, "proved")
    # a d_ext bound must never leak into a d_circ bracket
    assert bracket(Bound(2, "exact", quantity="d_ext", basis="Z"), basis="Z").lo is None


def test_min_over_bases_needs_both_lower_bounds():
    z = Bracket(5, 6, basis="Z")
    x_no_lower = Bracket(None, 7, basis="X")
    tot = min_over_bases(z, x_no_lower)
    assert tot.lo is None and tot.hi == 6
    assert min_over_bases(z, Bracket(4, 7, basis="X")).lo == 4


# ------------------------------------------------------------- circuits
def test_detectors_do_not_cancel_in_a_single_round():
    """A single noisy round needs two detector families per check: one closing
    the round-1 syndrome against the reset, one closing it against the final
    data readout.  Folded into a single ancilla-XOR-data detector the two
    cancel -- a data fault before the round flips the ancilla and the readout
    alike -- so the detector stays silent while the observable flips and every
    circuit reports ``d_circ = 1``.
    """
    code, s = _bb72()
    circ = memory_circuit(code, s.times, s.graph, basis="Z", p=1e-3)
    assert circ.num_detectors == 2 * code.rz
    _c, D, O = dem(code, s.times, s.graph, "Z")
    b = exact_cpsat(D, O, time_limit=120, ub=2)
    assert b is not None and b.kind == "lower" and b.value == 3


def test_deinterleave_preserves_per_check_order_and_is_proper():
    code, s = _bb72()
    ni = deinterleave(code, s.times, s.graph)
    by = {}
    for e, (ck, _q, _t) in enumerate(s.graph.edges):
        by.setdefault(ck, []).append(e)
    for es in by.values():
        assert sorted(es, key=lambda e: s.times[e]) == sorted(es, key=lambda e: ni[e])
    rep = Schedule(code=code, graph=s.graph, times=ni, T=int(ni.max()) + 1).verify()
    assert rep["proper"] and rep["contam"] and rep["sim"]


def test_stim_upper_bound_matches_certified_value():
    code, s = _bb72()
    u = upper_stim(code, s.times, s.graph, "Z", lim=4)
    assert u.kind == "upper" and u.value == 6        # certified d_circ(Z) = 6


def test_ext_distance_equals_certified_deinterleaved_distance():
    """Strikis, Browne and Beverland (arXiv:2603.05481) Thm 1: d_ext = d'_circ.
    Certified d'_circ(Z) = 3 on [[72,12,6]], and it must sit below the
    certified interleaved d_circ(Z) = 6."""
    code, s = _bb72()
    lo = lower_ext(code, s.times, s.graph, "Z", time_limit=120)
    assert lo is not None and lo.kind == "lower" and lo.quantity == "d_circ"
    assert lo.value == 3
    assert bracket(lo, upper_stim(code, s.times, s.graph, "Z", lim=4), basis="Z").hi == 6


# ------------------------------------------------------------- builders
SMALL = ["[[72,12,6]]", "[[90,8,10]]", "[[108,8,10]]", "[[144,12,12]]"]


def test_bb_catalogue_parameters_are_recomputed():
    for tag in SMALL:
        n, k, _d = (int(v) for v in tag.strip("[]").split(","))
        code = from_catalogue(tag)
        assert (code.n, code.k) == (n, k)


def test_sandwich_schedule_is_proper_at_delta_plus_one():
    for tag in SMALL:
        l, m, a, b = CATALOGUE[tag]
        code = bb_code(l, m, a, b, name=tag)
        s = bb_sandwich_schedule(code, l, m, a, b)
        rep = s.verify()
        assert rep["proper"] and rep["contam"] and rep["sim"]
        assert s.depth == code.delta + 1


def test_bbtile_reduced_model_and_nogoods():
    """The tiled sandwich model: depth 6 infeasible, depth 7 feasible and
    verified, and a motif's no-good has one translate per colour shift."""
    from autext.bb import CATALOGUE
    from autext.circ.bbtile import Tiling, linear_colouring

    l, m, a, b = CATALOGUE["[[144,12,12]]"]
    til = Tiling(l, m, a, b, colouring=linear_colouring(l, m, [(1, 1, 3)]))
    assert len(til.classes["X"]) == 3 and len(til.check_overlaps["X"]) == 432
    st, TX, TZ = til.reduced(6, 2, [], tl=30, workers=2)
    assert st == "INFEASIBLE" and TX is None
    st, TX, TZ = til.reduced(7, 2, [], tl=30, workers=2)
    assert TX is not None
    rep = til.verify(TX, TZ, 7)
    assert rep["proper"] and rep["contam"] and rep["sim"] and rep["depth"] == 7
    cols = til.residual_columns("Z", TX, TZ)
    assert len(cols) == 3 * 72
    used = [cols[0]]
    ngs = til.nogoods_from("Z", used)
    assert len(ngs) == 3                       # 72 translates collapse to the 3 colour shifts
    st2, TX2, _ = til.reduced(7, 2, ngs, tl=30, workers=2)
    if TX2 is not None:                        # the no-good is respected
        c = til.colour["X"][cols[0][1]]
        order = sorted(range(6), key=lambda t: TX2[c][t])
        assert tuple(sorted(order[:2])) != cols[0][3]


def test_cover_search_matches_the_certified_72_12_6_distance():
    """Cover branching reproduces the CP-SAT-certified d_circ = 6, and the
    orbit seeding is sound: all-mechanism seeds give the same answer."""
    import numpy as np

    from autext.circ import cover_ladder, dem, preserves_observables
    from autext.circ.symmetry import bb_detector_perms, column_perm

    tag = "[[72,12,6]]"
    l, m, a, b = CATALOGUE[tag]
    code = bb_code(l, m, a, b, name=tag)
    s = bb_sandwich_schedule(code, l, m, a, b)
    got = {}
    for basis in ("Z", "X"):
        _c, D, O = dem(code, s.times, s.graph, basis)
        rc = code.rz if basis == "Z" else code.rx
        perms = [column_perm(D, dp) for dp in bb_detector_perms(l, m, rc)]
        assert all(p is not None for p in perms)
        assert all(preserves_observables(D, O, p) for p in perms)
        from autext.circ.symmetry import orbit_reps
        reps = orbit_reps(D.shape[1], perms)
        w, _seed = cover_ladder(D, O, reps, 7, log=lambda _m: None)
        w_all, _ = cover_ladder(D, O, list(range(D.shape[1])), 7, log=lambda _m: None)
        assert w == w_all, (basis, w, w_all)
        got[basis] = w
    assert min(got.values()) == 6, got


def test_find_leq_upper_bound_mode_agrees_with_the_exact_distance():
    """``find_leq`` is the screen's cheap upper bound: a hit is a verified
    fault set of weight <= cap.  It must hit for every cap >= d_circ and miss
    for every cap < d_circ, and each witness must satisfy ``D v = 0``,
    ``O v != 0``.

    The mode exists because the binary question "is d_circ <= w?" does not
    need the ladder below w.  It is UNSOUND as a refutation -- a miss proves
    nothing -- so the only thing pinned here is the direction that is claimed.
    """
    import numpy as np

    from autext.circ import dem
    from autext.circ.cover import CoverSearch, orbit_seeds
    from autext.circ.symmetry import bb_detector_perms, column_perm

    tag = "[[72,12,6]]"
    l, m, a, b = CATALOGUE[tag]
    code = bb_code(l, m, a, b, name=tag)
    s = bb_sandwich_schedule(code, l, m, a, b)
    for basis in ("Z", "X"):
        _c, D, O = dem(code, s.times, s.graph, basis)
        rc = code.rz if basis == "Z" else code.rx
        perms = [column_perm(D, dp) for dp in bb_detector_perms(l, m, rc)]
        seeds = orbit_seeds(D.shape[1], perms)

        exact = None                       # the ladder, from w = 1
        for w in range(1, 9):
            cs = CoverSearch(D, O)
            if any(_seed_hit(cs, r, excl, w) for r, excl in seeds):
                exact = w
                break
        assert exact is not None, basis

        for cap in range(1, 9):
            cs = CoverSearch(D, O)
            cs.find_leq = True
            wit = None
            for r, excl in seeds:
                cs.exclude = excl
                if cs.seed(r, cap):
                    wit = list(cs.witness)
                    break
            if cap < exact:
                assert wit is None, (basis, cap, exact)
            else:
                assert wit is not None, (basis, cap, exact)
                v = np.zeros(D.shape[1], dtype=np.uint8)
                v[sorted(set(wit))] = 1
                assert not ((D @ v) % 2).any()
                assert bool(((O @ v) % 2).any())
                assert int(v.sum()) <= cap


def _seed_hit(cs, r, excl, w):
    cs.exclude = excl
    return cs.seed(r, w)


def test_cover_search_incremental_counts_stay_in_sync():
    """``_pick`` reads a per-detector count of usable mechanisms that the branch
    loop maintains incrementally instead of recounting from scratch.  A count
    that drifts would change which detector is branched on, and a refutation
    built on a changed enumeration is not a refutation.  So check the
    maintained counts against a direct recount at every node of a small search.
    """
    from autext.circ import dem
    from autext.circ.cover import CoverSearch
    from autext.circ.symmetry import bb_detector_perms, column_perm

    tag = "[[72,12,6]]"
    l, m, a, b = CATALOGUE[tag]
    code = bb_code(l, m, a, b, name=tag)
    s = bb_sandwich_schedule(code, l, m, a, b)
    _c, D, O = dem(code, s.times, s.graph, "Z")
    rc = code.rz
    assert all(column_perm(D, dp) is not None
               for dp in bb_detector_perms(l, m, rc))

    cs = CoverSearch(D, O)
    checked = [0]
    orig = cs._pick

    def spy(syn):
        checked[0] += 1
        if checked[0] <= 400:            # a sample is enough and keeps it quick
            forb = cs.forb
            for d in range(cs.nD):
                want = sum(1 for mm in cs.inc[d] if not forb[mm])
                assert cs.cnt[d] == want, (d, cs.cnt[d], want)
        return orig(syn)

    cs._pick = spy
    cs.seed(0, 5)
    assert checked[0] > 0
