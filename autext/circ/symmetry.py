"""Symmetry of the DEM, used as a cut in the exact solver.

A minimum-weight fault set contains *some* mechanism. If the circuit has an
automorphism group acting on the mechanisms, translate the set so that
mechanism is an orbit representative; hence some translate of every solution
satisfies ``sum_{m in reps} f_m >= 1``. That single cut breaks the symmetry
the branch-and-bound would otherwise thrash on, without changing the proved
answer. Pinning ``f_m = 1`` in ``|reps|`` separate subproblems is not the same
thing: it drops the objective's incumbent pruning.

For a BB code the group is the translation group ``Z_l x Z_m``, provided the
SCHEDULE is translation-invariant too --- the Eq. (S24) sandwich is, because
its edge colour is the monomial index. The detector permutation follows from
the check translation because :func:`autext.circ.build.memory_circuit` emits
detectors family-major; the column permutation follows from that because
detector signatures are unique per column, which :func:`column_perm` verifies
rather than assumes.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bb_detector_perms", "column_perm", "orbit_reps", "bb_orbit_cut",
           "preserves_observables"]


def bb_detector_perms(l: int, m: int, r_check: int, families: int = 2):
    """Detector permutations induced by every translation of ``Z_l x Z_m``."""
    out = []
    for da in range(l):
        for db in range(m):
            p = np.empty(families * r_check, dtype=np.int64)
            for c in range(r_check):
                i, j = divmod(c, m)
                c2 = ((i + da) % l) * m + (j + db) % m
                for t in range(families):
                    p[t * r_check + c] = t * r_check + c2
            out.append(p)
    return out


def column_perm(D, det_perm):
    """Lift a detector permutation to the DEM columns; ``None`` if it is not a
    DEM automorphism or the detector signatures are not unique. Returning
    ``None`` is the guard that keeps a wrong group from producing a wrong cut."""
    E = D.shape[1]
    sig = {}
    for e in range(E):
        k = tuple(np.nonzero(D[:, e])[0])
        if k in sig:
            return None
        sig[k] = e
    out = np.empty(E, dtype=np.int64)
    for e in range(E):
        k = tuple(sorted(det_perm[x] for x in np.nonzero(D[:, e])[0]))
        if k not in sig:
            return None
        out[e] = sig[k]
    return out


def preserves_observables(D, O, perm) -> bool:
    """Does the column permutation keep "this fault set flips a logical" true?

    The orbit argument needs the symmetry to map SOLUTIONS to solutions, not
    merely to preserve the detectors. That holds iff ``ker([D; O])`` is
    invariant, iff every row of the permuted ``O`` already lies in the row
    space of ``[D; O]`` --- a rank test. ``column_perm`` checks only the
    detector half, so a caller relying on the cut for a PROOF should call this
    too. (For a BB translation it is true on paper, since a translation is
    invertible on the logical quotient; this is the machine check of that.)
    """
    from ..linalg import rank_gf2

    D = np.asarray(D, dtype=np.uint8)
    O = np.asarray(O, dtype=np.uint8)
    base = np.vstack([D, O])
    r0 = rank_gf2(base)
    return rank_gf2(np.vstack([base, O[:, perm]])) == r0


def orbit_reps(E: int, perms):
    """One representative column per orbit."""
    seen = np.zeros(E, dtype=bool)
    reps = []
    for e in range(E):
        if seen[e]:
            continue
        reps.append(e)
        stack = [e]
        seen[e] = True
        while stack:
            x = stack.pop()
            for p in perms:
                y = int(p[x])
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
    return reps


def bb_orbit_cut(D, l: int, m: int, r_check: int, O=None, families: int | None = None):
    """Orbit representatives for a BB code's translation group, or ``None`` if
    the DEM does not in fact carry the symmetry (then no cut is applied).

    With ``O`` given, every translation is additionally required to preserve
    the logical condition (:func:`preserves_observables`), which is what the
    orbit argument needs when the result is used as a PROOF rather than as a
    search heuristic.  ``families`` is the number of detector blocks
    (``rounds + 1``); it defaults to ``D.shape[0] // r_check``, so a
    multi-round DEM works without being told.
    """
    if families is None:
        families = D.shape[0] // r_check      # rounds + 1 detector blocks
    perms = []
    for dp in bb_detector_perms(l, m, r_check, families):
        cp = column_perm(D, dp)
        if cp is None:
            return None
        if O is not None and not preserves_observables(D, O, cp):
            return None
        perms.append(cp)
    return orbit_reps(D.shape[1], perms)
