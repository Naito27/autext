"""Circuit distance of syndrome-extraction circuits, and the search for better
ones.

Layers, each importing only from those above it:

    build       one noisy round -> stim circuit -> (D, O);  sector()  fixes
                conventions
    bounds      typed Bound / Bracket; upper_stim, upper_bposd, lower_ext,
                exact_cpsat
    residual    tails, residual distances, the ranking proxy of Strikis,
                Browne and Beverland (arXiv:2603.05481)
    symmetry    BB translation automorphisms -> the orbit cut
    candidates  bb_schedules, qt_schedules, qt_construct
    search      the staged pipeline
    bench       command-line entry points

The one rule: a number about d_circ is a :class:`Bound` with a direction, never
a bare int, and a value is only "certified" when a lower bound meets an upper
bound in a :class:`Bracket` --- in every memory basis.
"""
from .bounds import (Bound, Bracket, bracket, exact_cpsat, ext_distance, lower_ext,
                     min_over_bases, upper_bposd, upper_stim)
from .build import (Sector, deinterleave, dem, dem_matrices, drop_dead, logical_basis,
                    memory_circuit, sector)
from .candidates import bb_schedules, qt_construct, qt_load, qt_schedules
from .residual import (cost_key, cost_vector, extended_matrices, rank, residual_dict,
                       residual_distance, tail_matrix, tails)
from .local import local_search, witness
from .cover import CoverSearch, cover_ladder
from .search import pipeline, screen, stim_parallel
from .symmetry import bb_orbit_cut, orbit_reps, preserves_observables

__all__ = [
    "Bound", "Bracket", "bracket", "min_over_bases",
    "upper_stim", "upper_bposd", "lower_ext", "ext_distance", "exact_cpsat",
    "Sector", "sector", "logical_basis", "memory_circuit", "deinterleave",
    "dem", "dem_matrices", "drop_dead",
    "tails", "tail_matrix", "extended_matrices", "residual_distance",
    "residual_dict", "cost_vector", "cost_key", "rank",
    "bb_orbit_cut", "orbit_reps", "preserves_observables",
    "CoverSearch", "cover_ladder",
    "bb_schedules", "qt_load", "qt_schedules", "qt_construct",
    "screen", "pipeline", "stim_parallel", "local_search", "witness",
]
