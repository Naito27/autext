"""Left-right circuits of Strikis, Browne and Beverland (arXiv:2603.05481).

A left-right circuit (LRC) is a NON-interleaving syndrome-extraction circuit: it
partitions the data qubits into a left set and a right set, inducing

    H_X = [ L_X | R_X ],    H_Z = [ L_Z | R_Z ],

and staggers the X and Z checks so that the left half of the X-checks runs
alongside the right half of the Z-checks, then the other two.  With `delta(.)`
the maximum degree of a bipartite subgraph -- equal to its chromatic index by
Koenig, so it is exactly the number of CNOT layers that block needs -- their
Eq. (6) gives the depth

    t = t_1 + t_2 + 2,   t_1 = max(delta(L_X), delta(R_Z)),
                         t_2 = max(delta(L_Z), delta(R_X)),

where the `+2` is the ancilla preparation and measurement steps.  The AMORTIZED
cost over many rounds drops those two (consecutive rounds stagger), leaving
`t_1 + t_2` per round, which is what `lrc_cost` returns and what the comparison
here uses.

**`t_1 + t_2 >= Delta` for every partition.**  Take any qubit `q`.  If `q` is on
the left it contributes `deg_X(q)` to `delta(L_X)` and `deg_Z(q)` to
`delta(L_Z)`, so `t_1 >= deg_X(q)` and `t_2 >= deg_Z(q)`; if it is on the right
it contributes `deg_Z(q)` to `delta(R_Z)` and `deg_X(q)` to `delta(R_X)`, giving
the same two bounds with the roles swapped.  Either way
`t_1 + t_2 >= deg_X(q) + deg_Z(q)`.  Similarly a check of weight `w` splits as
`|supp cap L| + |supp cap R|` with the two parts bounded by `t_1` and `t_2`, so
`t_1 + t_2 >= w`.  Taking the maximum over qubits and checks gives
`t_1 + t_2 >= Delta` -- the same Koenig bound an interleaved circuit is measured
against.  An interleaved circuit of depth `Delta` therefore matches or beats the
amortized cost of ANY left-right circuit for the same code; the open question is
only whether a partition attains `Delta`, which `best_lrc` searches for.
"""
import numpy as np

__all__ = ["lrc_cost", "best_lrc", "lrc_bound"]


def _delta(M, cols):
    """Maximum degree of the bipartite subgraph on the selected columns."""
    if not cols.any():
        return 0
    sub = M[:, cols]
    return max(int(sub.sum(1).max(initial=0)), int(sub.sum(0).max(initial=0)))


def lrc_cost(HX, HZ, left):
    """`(t1 + t2, t1, t2)` for the qubit partition `left` (a boolean mask)."""
    left = np.asarray(left, bool)
    right = ~left
    t1 = max(_delta(HX, left), _delta(HZ, right))
    t2 = max(_delta(HZ, left), _delta(HX, right))
    return t1 + t2, t1, t2


def lrc_bound(HX, HZ):
    """`Delta`, the lower bound on `t1 + t2` over all partitions (see module doc)."""
    return int(max(HX.sum(1).max(initial=0), HZ.sum(1).max(initial=0),
                   (HX.sum(0) + HZ.sum(0)).max(initial=0)))


def best_lrc(HX, HZ, restarts=12, sweeps=40, seed=0, seeds_extra=(),
             cell_of=None):
    """Search qubit partitions for the cheapest amortized left-right cost.

    Greedy descent on single-qubit flips from random and structured starts.
    Returns `(best_cost, best_left_mask, bound)`; stops early if the bound
    `Delta` is attained, since nothing can do better.

    `cell_of` restricts the search to partitions CONSTANT ON FIBRES: pass the
    array sending each lifted qubit `(i, j, g)` to its base square `(i, j)` and
    the flips happen per square, so what is searched is a partition of the base
    graph.  On a lifted code that loses nothing: a base partition lifts to a
    partition of the lifted qubits with the same `t1 + t2`, and the base graph
    is small enough for the descent to reach the optimum, where the same search
    over all `n` lifted qubits can get stuck.
    """
    n = HX.shape[1]
    cell_of = None if cell_of is None else np.asarray(cell_of)
    ncell = n if cell_of is None else int(cell_of.max()) + 1

    def expand(cm):
        return cm if cell_of is None else cm[cell_of]
    bound = lrc_bound(HX, HZ)
    rng = np.random.default_rng(seed)
    dX, dZ = HX.sum(0), HZ.sum(0)
    if cell_of is not None:      # per-cell degrees, for the degree-aware starts
        dX = np.bincount(cell_of, weights=dX, minlength=ncell)
        dZ = np.bincount(cell_of, weights=dZ, minlength=ncell)
    starts = [np.zeros(ncell, bool)]
    starts[0][: ncell // 2] = True
    starts.append(np.arange(ncell) % 2 == 0)
    # degree-aware splits: a left qubit charges deg_X to t_1 and deg_Z to t_2,
    # a right one the other way round, so separating the X-heavy from the
    # Z-heavy qubits is what balances the two halves.
    starts.append(dX <= dZ)
    starts.append(dX > dZ)
    starts += [np.asarray(s, bool) for s in seeds_extra]
    starts += [rng.random(ncell) < 0.5 for _ in range(restarts)]

    best, best_mask = None, None
    for mask in starts:
        cur = mask.copy()
        cost = lrc_cost(HX, HZ, expand(cur))[0]
        for _ in range(sweeps):
            improved = False
            for q in rng.permutation(ncell):
                cur[q] = ~cur[q]
                c = lrc_cost(HX, HZ, expand(cur))[0]
                if c < cost:
                    cost = c
                    improved = True
                else:
                    cur[q] = ~cur[q]
            if not improved:
                break
        if best is None or cost < best:
            best, best_mask = cost, expand(cur).copy()
        if best == bound:
            break
    return best, best_mask, bound
