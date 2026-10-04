"""Quantum Tanner codes: depth-optimal parity-check circuits.

Numerics for the QT sections of *Disassembling qLDPC codes for depth-optimal
parity-check circuits*. The QT code is assembled as a tensor product of four
classical codes followed by a group lift; inverting the lift (the edge partition
``P_lift``) leaves the **base Tanner graph** on an ``nA x nB`` grid of squares,
which is what everything here schedules.

- :mod:`~autext.qt.qt`      -- families, quadrants, ``roles(q)``, halvings, ``QT``
- :mod:`~autext.qt.build`   -- build a scheduling instance from four matrix pairs
- :mod:`~autext.qt.core`    -- ``Inst``, ``verify``, ``straddles``, ``layer_hall``
- :mod:`~autext.qt.simple`  -- the layer-sweep solver (the paper's Box 1)
- :mod:`~autext.qt.exact`   -- CP-SAT feasibility at fixed depth
- :mod:`~autext.qt.lift`    -- build the *lifted* QT code and lift a base schedule
- :mod:`~autext.qt.stress`  -- the stress battery (all regimes)

Submodules are imported on demand (``from autext.qt import qt, build``) rather
than eagerly, so that ``python -m autext.qt.<name>`` runs a module exactly once.
"""

__all__ = ["qt", "codes", "build", "core", "simple", "exact", "group", "lift"]
