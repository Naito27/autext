# autext — depth-optimal parity-check circuits for qLDPC codes

Companion code to ***Disassembling qLDPC codes for depth-optimal parity-check
circuits*** (Nguyen, Rimbach-Russ, Bosco).

Many good qLDPC codes are *assembled* from small components by explicit
operations — tensor products, group lifts — that imprint edge symmetries on the
Tanner graph. Instead of scheduling the full code, we **disassemble** it: quotient
those symmetries with a sequence of edge partitions `P_i`, solve the much smaller
reduced scheduling problem, and lift the circuit back.

The focus of this repository is the **quantum Tanner (QT)** family, where the
reduced problem is solved numerically.

```bash
pip install -e .            # numpy only
pip install -e ".[all]"     # + ortools, networkx, stim, ldpc, pytest
pytest -q                   # 55 tests
```

The extras are independent: `solve` (ortools) for the exact fixed-depth
feasibility solver, `analysis` (networkx) for the orientation-validity script,
`stim` for independent stabilizer-circuit validation, and `circ` (stim + ldpc)
for the circuit-level distance work.

## Quick start

```python
from autext.qt import qt, build, core, simple

Q = qt.QT('[6,3,3]', '[6,3,3]')                 # base code; pi = id, assumption [A]
# Q = qt.QT.random('[6,3,3]', '[6,3,3]', rng)   # the general coupled relabelling
rho   = qt.weight_class_halving(Q.H)            # row halving
sigma = qt.weight_class_halving(Q.Hp)           # column halving
inst  = build.from_qt(Q, rho, sigma)            # base Tanner graph + sandwich groups
T, sch = simple.solve(inst)                     # the layer sweep (paper's Box 1)
core.verify(inst, sch, T)                       # never trust an unverified schedule
```

and to carry that reduced schedule back to the *actual* lifted code:

```python
from autext.qt.group import Group
from autext.qt.lift import lift_and_verify

G = Group.dihedral(3)
out = lift_and_verify(Q.pair, Q.nA, Q.nB, G,
                      A=[0]*Q.nA, B=[1]*Q.nB, base_sched=sch, T=T)
out['is_css'], out['depth'], out['report']
# True, 10, {'proper': True, 'contam': True, 'sim': True, ...}
```

## Layout

| path | what it is |
|---|---|
| `autext/qt/` | **the QT work** — base graph, solvers, result checks |
| `autext/qt/qt.py` | families, quadrants, the sandwich roles, and the base code of Eq. (S45) |
| `autext/qt/simple.py` | the layer sweep of the paper's Box 1 |
| `autext/qt/exact.py` | CP-SAT feasibility at fixed depth — separates a hard obstruction from a greedy failure |
| `autext/qt/core.py` | the instance type, the verifier, the straddling test, the window lemma |
| `autext/qt/lift.py` | builds the *lifted* code (Eq. S48) and transports a base schedule onto it |
| `autext/qt/group.py` | minimal finite groups for the lift (`cyclic`, `dihedral`, `from_perms`) |
| `autext/qt/stress.py` | the stress battery: `python -m autext.qt.stress <scale\|aspect\|rates\|expander\|exsweep\|general>` |
| `autext/qt/sweep_simple_base.py` | the base-pair sweep — no argument for `pi = id`, seeds for `pi != id` |
| `autext/qt/tables.py` | the two manuscript tables: `python -m autext.qt.tables base\|lift` |
| `autext/circ/` | **circuit-level distance** of a schedule — see below |
| `autext/core.py` | `CSSCode`, `TannerGraph` — the code-agnostic data model |
| `autext/schedule.py`, `autext/validation.py` | `Schedule` and its verifiers (parity formula, propagation simulation, stim) |
| `autext/baselines.py` | staggered upper bound and the full CP-SAT optimum of Eq. (3) |
| `autext/bb.py`, `autext/lp_sandwich.py` | bivariate-bicycle codes and their sandwich schedules |
| `autext/linalg.py`, `autext/constraints.py`, `autext/distance.py` | GF(2), edge colouring, code distance |
| `results/` | the LaTeX manuscript tables |
| `tests/` | the regression suite |

## Circuit-level distance

`autext/circ/` computes the circuit-level distance `d_circ` of a schedule: the
minimum number of circuit faults producing an undetectable logical error. A
number about `d_circ` is never a bare integer — `circ/bounds.py` returns a
`Bound` carrying its direction (upper, lower or exact), so an upper bound cannot
be mistaken for a lower one.

| module | role |
|---|---|
| `circ/build.py` | schedule → stim circuit → detector error model `(D, O)` |
| `circ/bounds.py` | stim's search, a BP-OSD estimator, the extended-code lower bound, exact CP-SAT |
| `circ/cover.py` | exhaustive minimum-weight search by cover branching on the detector error model |
| `circ/symmetry.py` | translation automorphisms of bivariate-bicycle codes, used to cut the search |
| `circ/residual.py` | residual (hook) errors of a check order, and ranking by them |
| `circ/candidates.py`, `circ/search.py`, `circ/local.py` | candidate schedules, the screening pipeline, local search |
| `circ/bench.py` | CLI: `python -m autext.circ.bench validate\|certify\|optimise\|construct` |

`python -m autext.circ.bench validate` is the oracle check: on the surface code,
stim, CP-SAT and BP-OSD must agree.

## The two verification layers

Nothing is reported that has not been machine-checked, at whichever level it lives:

- **base graph** — `autext.qt.core.verify(inst, sch, T)` checks the layer range,
  both collision conditions, and the per-square group order of the sandwich.
- **full code** — `autext.qt.lift.lift_and_verify(...)` builds the lifted code,
  inverts `P_lift`, and runs `Schedule.verify()`: properness by the Eq. (2)
  parity formula *and* an independent propagation simulation (plus a stim
  tableau check when `stim` is installed).

The second layer is what closes the paper's argument: a depth-Δ schedule found on
the reduced problem lifts to a proper, depth-Δ circuit on the real QT code.

## Conventions worth knowing before reading the code

- A QT code's four families need not share matrices. The general base code takes
  two independent local codes per side; `qt.pairs_from_codes` is the single place
  a family-to-matrix map is written, and every builder asserts `qt.commutes` on
  its output, so what gets scheduled is always a genuine CSS code.
- No structural assumptions are made about the local codes: no systematic form,
  no complementarity, no regularity, no constant column weights. Degenerate
  (zero-weight) columns are allowed and do occur.
- `core.straddles` is a *necessary* condition for depth Δ at a given halving. A
  halving that fails it cannot reach Δ, which is a structural fact about the
  instance rather than a failure of the solver.
