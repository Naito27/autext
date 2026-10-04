"""The :class:`Schedule` object: a depth-``T`` assignment of layers to CX gates.

Every scheduler returns one of these. It bundles the per-edge layer assignment
with the code/graph it belongs to and exposes the full verification stack:
the (C1) proper-colouring check, the (C2) contamination parity formula, an
independent X-propagation simulation, and (optionally) a stim tableau check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constraints import contamination_ok, verify_proper
from .core import CSSCode, TannerGraph

__all__ = ["Schedule"]


@dataclass
class Schedule:
    """A scheduled syndrome-extraction circuit."""

    code: CSSCode
    graph: TannerGraph
    times: np.ndarray  # layer per edge id
    T: int             # layer count the schedule was solved at
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=int)

    @property
    def depth(self) -> int:
        """Actual depth = number of distinct layers used."""
        return int(self.times.max(initial=-1)) + 1

    # -- verification ----------------------------------------------------
    def verify_proper(self) -> bool:
        """(C1) proper edge colouring within ``T`` layers."""
        return verify_proper(self.graph.edges, self.times, self.T)

    def verify_contamination(self) -> bool:
        """(C2) net-leak parity even on every overlapping pair (formula)."""
        return contamination_ok(self.times, self.graph)

    def simulate(self) -> tuple[bool, list[tuple[int, int]]]:
        """Independent check: propagate X off each X-ancilla through the circuit.

        Returns ``(ok, bad_pairs)``; ``ok`` is True when no X lands on a
        Z-ancilla. See :func:`autext.validation.simulate_contamination`.
        """
        from .validation import simulate_contamination

        return simulate_contamination(self.code, self.times, self.graph)

    def stim_check(self) -> tuple[bool, bool]:
        """stim tableau check: ``(stabilizers_invariant, no_ancilla_leak)``.

        Requires the optional ``stim`` dependency.
        """
        from .validation import stim_validate

        return stim_validate(self.code, self.times, self.graph)

    def verify(self, stim: bool = False) -> dict:
        """Run all available checks and return a results dict.

        ``proper`` and ``contam`` use the parity formula; ``sim`` is the
        independent X-propagation; ``stim`` (optional) the tableau check.
        """
        out = {
            "depth": self.depth,
            "proper": self.verify_proper(),
            "contam": self.verify_contamination(),
        }
        sim_ok, bad = self.simulate()
        out["sim"] = sim_ok
        out["sim_bad"] = bad
        if stim:
            inv_ok, leak_ok = self.stim_check()
            out["stim_invariant"] = inv_ok
            out["stim_no_leak"] = leak_ok
        return out

    def is_valid(self) -> bool:
        """True iff proper, contamination-free (formula), and sim all agree."""
        sim_ok, _ = self.simulate()
        return self.verify_proper() and self.verify_contamination() and sim_ok
