# autext

Code for the paper *Disassembling qLDPC codes for depth-optimal parity-check
circuits* (Nguyen, Rimbach-Russ, Bosco).

Many good qLDPC codes are built from small pieces by tensor products and group
lifts, and these constructions leave symmetries in the code's Tanner graph. We
use those symmetries to shrink the scheduling problem. A sequence of edge
partitions `P_i` folds the Tanner graph into a much smaller one, we find a
parity-check schedule for the small graph, and then we lift that schedule back
to the full code.

This repository does this for quantum Tanner (QT) codes, where the small
problem is solved numerically.

## Installation

```bash
pip install -e .           # the core package; needs only numpy
pip install -e ".[all]"    # all optional dependencies, plus pytest
pytest -q                  # run the test suite
```

You can also install the optional dependencies one group at a time:

| extra | installs | needed for |
|---|---|---|
| `solve` | ortools | the exact solver at fixed depth (`autext/qt/exact.py`) and the other CP-SAT models |
| `analysis` | networkx | the orientation check in `autext/qt/orient.py` |
| `stim` | stim | an independent stim check of lifted circuits |
| `circ` | stim, ldpc | the circuit-level distance code in `autext/circ/` |

## Example

Build a small QT code and schedule its base graph:

```python
from autext.qt import qt, build, core, simple

Q = qt.QT('[6,3,3]', '[6,3,3]')                 # base code with pi = id (assumption [A])
# Q = qt.QT.random('[6,3,3]', '[6,3,3]', rng)   # or a random coupled relabelling
rho   = qt.weight_class_halving(Q.H)            # split the rows in two
sigma = qt.weight_class_halving(Q.Hp)           # split the columns in two
inst  = build.from_qt(Q, rho, sigma)            # base Tanner graph and sandwich groups
T, sch = simple.solve(inst)                     # the layer sweep of the paper's Box 1
core.verify(inst, sch, T)                       # raises if the schedule is invalid
```

Then lift the schedule to the full code and check it there:

```python
from autext.qt.group import Group
from autext.qt.lift import lift_and_verify

G = Group.dihedral(3)
out = lift_and_verify(Q.pair, Q.nA, Q.nB, G,
                      A=[0]*Q.nA, B=[1]*Q.nB, base_sched=sch, T=T)
out['is_css'], out['depth'], out['report']
# True, 10, {'proper': True, 'contam': True, 'sim': True, ...}
```

## How schedules are checked

We only report a schedule after a verifier has accepted it. The checks run at
two levels.

1. **On the base graph.** `autext.qt.core.verify(inst, sch, T)` checks that
   every edge is in a layer between 1 and `T`, that neither collision condition
   is violated, and that the sandwich groups at each square come in the right
   order.
2. **On the full code.** `autext.qt.lift.lift_and_verify(...)` builds the lifted
   code, carries the schedule over to it by inverting `P_lift`, and runs
   `Schedule.verify()`. That checks the circuit is proper in two independent
   ways, with the parity formula of Eq. (2) and with a propagation simulation.
   If stim is installed, it also runs a stim tableau check.

The paper's argument depends on the second level, because it shows that a
depth-Δ schedule found on the small graph gives a proper depth-Δ circuit on the
actual QT code.

## Things to know before reading the code

- The four families of a QT code don't have to use the same matrices. The
  general base code takes two independent local codes on each side.
  `qt.pairs_from_codes` is the only place where families are mapped to
  matrices, and every builder checks `qt.commutes` on its output, so whatever
  gets scheduled is a valid CSS code.
- We assume nothing about the structure of the local codes. They don't need to
  be in systematic form, complementary or regular, or to have constant column
  weight. Columns of weight zero are allowed, and they do occur.
- `core.straddles` is a necessary condition for reaching depth Δ with a given
  halving. If a halving fails it, no solver can reach Δ with that halving. This
  is a property of the instance, not a weakness of the solver.

## Repository layout

The QT work is in `autext/qt/`:

| file | contents |
|---|---|
| `qt.py` | code families, quadrants, sandwich roles, and the base code of Eq. (S45) |
| `simple.py` | the layer sweep of the paper's Box 1 |
| `exact.py` | an exact CP-SAT feasibility check at fixed depth, which tells a real obstruction apart from a failure of the greedy sweep |
| `core.py` | the instance type, the verifier, the straddling test and the window lemma |
| `lift.py` | builds the lifted code (Eq. S48) and carries a base schedule onto it |
| `group.py` | small finite groups to lift with (`cyclic`, `dihedral`, `from_perms`) |
| `stress.py` | stress tests: `python -m autext.qt.stress scale\|aspect\|rates\|expander\|exsweep\|general` |
| `sweep_simple_base.py` | the sweep over base pairs; run it with no argument for `pi = id`, or with seeds for `pi != id` |
| `tables.py` | recomputes the paper's two tables: `python -m autext.qt.tables base\|lift` |

The top level of `autext/` holds code that works for any CSS code:

| file | contents |
|---|---|
| `core.py` | the data model, `CSSCode` and `TannerGraph` |
| `schedule.py`, `validation.py` | `Schedule` and its verifiers (parity formula, propagation simulation, stim) |
| `baselines.py` | the staggered upper bound and the full CP-SAT optimum of Eq. (3) |
| `bb.py`, `lp_sandwich.py` | bivariate-bicycle codes and their sandwich schedules |
| `linalg.py`, `constraints.py`, `distance.py` | linear algebra over GF(2), edge colouring, code distance |

The tests are in `tests/`.

## Circuit-level distance

`autext/circ/` computes the circuit-level distance `d_circ` of a schedule. This
is the smallest number of faults in the circuit that cause a logical error
without being detected. The code never reports `d_circ` as a plain integer.
`circ/bounds.py` returns a `Bound` that records whether the number is an upper
bound, a lower bound or an exact value, so the two kinds of bound can't be
mixed up.

| file | contents |
|---|---|
| `build.py` | turns a schedule into a stim circuit and its detector error model `(D, O)` |
| `bounds.py` | stim's search, a BP-OSD estimate, the extended-code lower bound, and exact CP-SAT |
| `cover.py` | exhaustive minimum-weight search on the detector error model, by branching on covers |
| `symmetry.py` | translation symmetries of bivariate-bicycle codes, used to shrink the search |
| `residual.py` | the residual (hook) errors of a check order, and a ranking of orders by them |
| `candidates.py`, `search.py`, `local.py` | candidate schedules, the screening pipeline, and local search |
| `bench.py` | command line: `python -m autext.circ.bench validate\|certify\|optimise\|construct` |

To check the tools against each other, run `python -m autext.circ.bench
validate`. On the surface code, stim, CP-SAT and BP-OSD must all give the same
distance.
