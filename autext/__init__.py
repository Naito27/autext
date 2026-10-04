"""autext — depth-optimal parity-check circuits for qLDPC codes.

Companion code to *Disassembling qLDPC codes for depth-optimal parity-check
circuits* (Nguyen, Rimbach-Russ, Bosco).

Many good qLDPC codes are **assembled** from small components by explicit
operations (tensor products, group lifts) that imprint edge symmetries on the
Tanner graph. Rather than scheduling the full code, we **disassemble** it: quotient
those symmetries with a sequence of edge partitions ``P_i``, solve the small
reduced scheduling problem, and lift the circuit back.

The focus of this package is the **quantum Tanner (QT)** family, in
:mod:`autext.qt`:

    from autext.qt import qt, build, core, simple

    Q = qt.QT('[6,3,3]', '[6,3,3]')            # base code from two classical codes
    rho, sigma = qt.weight_class_halving(Q.H), qt.weight_class_halving(Q.Hp)
    inst = build.from_qt(Q, rho, sigma)        # base Tanner graph + sandwich groups
    T, sch = simple.solve(inst)                # layer sweep (the paper's Box 1)
    core.verify(inst, sch, T)                  # raises on any violation

The top level holds the code-agnostic pieces used to check a finished circuit on
the *full* code: :class:`CSSCode` / :class:`TannerGraph`, :class:`Schedule` and
its verifiers, the GF(2) helpers, and the two reference schedulers.
"""
from __future__ import annotations

from .baselines import exact_schedule, staggered_schedule
from .core import CSSCode, Edge, TannerGraph
from .distance import exact_distance, probabilistic_distance
from .linalg import nullspace_gf2, rank_gf2, rref_gf2
from .schedule import Schedule

__version__ = "0.2.0"

__all__ = [
    # core data model
    "CSSCode", "TannerGraph", "Edge", "Schedule",
    # reference schedulers
    "staggered_schedule", "exact_schedule",
    # GF(2)
    "rref_gf2", "rank_gf2", "nullspace_gf2",
    # code parameters
    "probabilistic_distance", "exact_distance",
]
