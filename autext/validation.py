"""Independent validators that cross-check the contamination parity formula.

- :func:`simulate_contamination` — push X off each X-ancilla and Z off each
  Z-ancilla through the CX circuit, and flag any that lands on an ancilla of the
  other type. Pure numpy, no parity formula, so a genuine cross-check of
  :func:`autext.constraints.contamination_ok`.

  On a CSS code with a proper schedule the X pass alone is sufficient: for each
  shared qubit exactly one of ``tau(a,q) < tau(b,q)``, ``tau(b,q) < tau(a,q)``
  holds, properness forbidding a tie, so ``Xleak + Zleak = |shared| (mod 2)``,
  and commutation makes ``|shared|`` even, hence ``Zleak == Xleak``.
- :func:`stim_validate` — stim Clifford-tableau check of (a) stabilizer
  invariance and (b) no ancilla leak. Requires the optional ``stim`` dependency.
"""
from __future__ import annotations

import numpy as np

from .core import CSSCode, TannerGraph

__all__ = ["simulate_contamination", "stim_validate", "build_stim_circuit"]


# --------------------------------------------------------------------------
# X-propagation simulation (numpy)
# --------------------------------------------------------------------------
def simulate_contamination(code: CSSCode, times, graph: TannerGraph | None = None):
    """Return ``(ok, bad_pairs)``; ``ok`` is True iff no X reaches a Z-ancilla."""
    graph = graph if graph is not None else TannerGraph.build(code)
    HX, HZ = code.HX, code.HZ
    n, rX, rZ = code.n, code.rx, code.rz
    W = n + rX + rZ

    def xanc(a):
        return n + a

    def zanc(b):
        return n + rX + b

    gates = []
    for e, (ck, q, ty) in enumerate(graph.edges):
        t = times[e]
        if ty == "X":
            gates.append((t, xanc(ck[1]), q))   # control = X-ancilla, target = qubit
        else:
            gates.append((t, q, zanc(ck[1])))   # control = qubit, target = Z-ancilla
    gates.sort(key=lambda g: g[0])

    # Both directions, each vectorised over its own ancillas: column j of X is
    # "where an X started on X-ancilla j has spread to", so the CX rule is one
    # row XOR and the gate loop runs once instead of once per ancilla.
    #
    # X direction: the X-ancilla is the CONTROL, so X spreads to the data and
    # reaches a Z-ancilla whose CX on a shared qubit fires later.
    X = np.zeros((W, rX), dtype=bool)
    X[[xanc(a) for a in range(rX)], np.arange(rX)] = True
    for (_t, c, tg) in gates:                       # X_c -> X_c X_t
        X[tg] ^= X[c]
    xleak = X[[zanc(b) for b in range(rZ)]] if rZ else np.zeros((0, rX), bool)
    bs, as_ = np.nonzero(xleak)
    bad = {(int(a), int(b)) for a, b in zip(as_, bs)}

    # Z direction: the Z-ancilla is the TARGET, and Z on a target propagates to
    # the control, so Z spreads back to the data and on to any X-ancilla whose
    # CX fires later.
    Z = np.zeros((W, rZ), dtype=bool)
    Z[[zanc(b) for b in range(rZ)], np.arange(rZ)] = True
    for (_t, c, tg) in gates:                       # Z_t -> Z_c Z_t
        Z[c] ^= Z[tg]
    zleak = Z[[xanc(a) for a in range(rX)]] if rX else np.zeros((0, rZ), bool)
    as_, bs = np.nonzero(zleak)
    bad |= {(int(a), int(b)) for a, b in zip(as_, bs)}

    return len(bad) == 0, sorted(bad)


# --------------------------------------------------------------------------
# stim tableau validation (optional)
# --------------------------------------------------------------------------
def build_stim_circuit(code: CSSCode, times, graph: TannerGraph | None = None):
    import stim

    graph = graph if graph is not None else TannerGraph.build(code)
    n, rX, rZ = code.n, code.rx, code.rz

    def xanc(a):
        return n + a

    def zanc(b):
        return n + rX + b

    gates = []
    for e, (ck, q, ty) in enumerate(graph.edges):
        t = times[e]
        if ty == "X":
            gates.append((t, xanc(ck[1]), q))
        else:
            gates.append((t, q, zanc(ck[1])))
    gates.sort(key=lambda g: g[0])

    c = stim.Circuit()
    for _t, ctrl, tgt in gates:
        c.append("CX", [ctrl, tgt])
    return c, n, rX, rZ


def _pauli(N, support, pauli):
    import stim

    p = stim.PauliString(N)
    code = 1 if pauli == "X" else 3
    for q in support:
        p[q] = code
    return p


def stim_validate(code: CSSCode, times, graph: TannerGraph | None = None):
    """Return ``(stabilizers_invariant, no_ancilla_leak)``. Requires ``stim``."""
    import stim

    c, n, rX, rZ = build_stim_circuit(code, times, graph)
    N = n + rX + rZ
    t = c.to_tableau()
    if len(t) < N:
        t = t + stim.Tableau(N - len(t))

    def xanc(a):
        return n + a

    def zanc(b):
        return n + rX + b

    HX, HZ = code.HX, code.HZ
    inv_ok = leak_ok = True

    for a in range(rX):
        supp = list(np.nonzero(HX[a])[0])
        P = _pauli(N, supp, "X")
        if t(P) != P:
            inv_ok = False
    for b in range(rZ):
        supp = list(np.nonzero(HZ[b])[0])
        P = _pauli(N, supp, "Z")
        if t(P) != P:
            inv_ok = False

    for a in range(rX):
        supp = list(np.nonzero(HX[a])[0])
        if t(_pauli(N, [xanc(a)], "X")) != _pauli(N, [xanc(a)] + supp, "X"):
            leak_ok = False
    for b in range(rZ):
        supp = list(np.nonzero(HZ[b])[0])
        if t(_pauli(N, [zanc(b)], "Z")) != _pauli(N, [zanc(b)] + supp, "Z"):
            leak_ok = False

    return inv_ok, leak_ok
