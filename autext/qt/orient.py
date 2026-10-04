"""Which row/column orientation pairs are chronologically valid?

A row constrains the two pairs {X0,Z1} and {X1,Z0}; a column constrains {Z1,X1}
and {Z0,X0}.  Those four pairs are exactly the edges of the 4-cycle

        X0 --(row)-- Z1 --(col)-- X1 --(row)-- Z0 --(col)-- X0

so at a data qubit (i,j) the constraint graph is C_4 with its two row edges
oriented by row i and its two column edges by column j.  A schedule exists iff
that orientation is acyclic.  The diagonals {X0,X1} and {Z0,Z1} are never
constrained -- no line couples two X's or two Z's.
"""
import itertools, networkx as nx

F = ['X0', 'Z1', 'X1', 'Z0']                 # cycle order
ROW_E = [('X0', 'Z1'), ('X1', 'Z0')]         # a, b   (forward = as written)
COL_E = [('Z1', 'X1'), ('Z0', 'X0')]         # c, d

def digraph(a, b, c, d):
    G = nx.DiGraph(); G.add_nodes_from(F)
    for (u, v), s in zip(ROW_E + COL_E, (a, b, c, d)):
        G.add_edge(*( (u, v) if s else (v, u) ))
    return G

def height(G):
    return nx.dag_longest_path_length(G) + 1

def shape(G):
    """number of antichain levels, and the level sizes"""
    lev = {}
    for n in nx.topological_sort(G):
        lev[n] = max((lev[p] + 1 for p in G.predecessors(n)), default=0)
    out = {}
    for n, l in lev.items():
        out.setdefault(l, []).append(n)
    return tuple(len(out[l]) for l in sorted(out))

if __name__ == '__main__':
    rows = {(a, b): ('alternating' if a != b else f'uniform s={a}') for a in (0,1) for b in (0,1)}
    cols = {(c, d): ('alternating' if c != d else f'uniform t={c}') for c in (0,1) for d in (0,1)}

    print(f"{'row (a,b)':>12s} {'type':>14s} | {'col (c,d)':>10s} {'type':>14s} | "
          f"{'acyclic':>7s} {'shape':>10s} {'height':>6s}")
    tally = {}
    for a, b in itertools.product((0,1), repeat=2):
        for c, d in itertools.product((0,1), repeat=2):
            G = digraph(a, b, c, d)
            ok = nx.is_directed_acyclic_graph(G)
            sh = shape(G) if ok else None
            tally[(rows[(a,b)].split()[0], cols[(c,d)].split()[0],
                   'same' if (a==b and c==d and a==c) else
                   ('opp' if (a==b and c==d) else '-'), ok, sh)] = \
                tally.get((rows[(a,b)].split()[0], cols[(c,d)].split()[0],
                   'same' if (a==b and c==d and a==c) else
                   ('opp' if (a==b and c==d) else '-'), ok, sh), 0) + 1
            print(f"{str((a,b)):>12s} {rows[(a,b)]:>14s} | {str((c,d)):>10s} "
                  f"{cols[(c,d)]:>14s} | {str(ok):>7s} {str(sh):>10s} "
                  f"{(height(G) if ok else '-'):>6}")

    print('\nsummary (row type, col type, sign, acyclic, level sizes) -> count')
    for k, v in sorted(tally.items(), key=lambda kv: str(kv[0])):
        print('  ', k, '->', v)

    # the invalid set, stated directly
    bad = [(a,b,c,d) for a,b,c,d in itertools.product((0,1),repeat=4)
           if not nx.is_directed_acyclic_graph(digraph(a,b,c,d))]
    print('\ninvalid (a,b,c,d):', bad)
    print('i.e. exactly a==b==c==d :', all(len({a,b,c,d})==1 for a,b,c,d in bad),
          ' and all such are invalid:',
          all(not nx.is_directed_acyclic_graph(digraph(x,x,x,x)) for x in (0,1)))
