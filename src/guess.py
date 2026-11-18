"""Guess a row's goals from the training rows that fall into the same cluster-tree leaf."""
from cluster import grow, leaf


def fit(t, rows, rng, stop=32):
    return grow(t, rows, rng, stop)


def guess(t, tree, row, lo=0.1, hi=0.9):
    """For each goal, (mean, low, high) over the leaf's rows; [low, high] says how sure."""
    rows, out = leaf(t, tree, row).rows, []
    for c in t.y:
        v = sorted(r.cells[c.at] for r in rows if r.cells[c.at] != "?")
        out.append((sum(v) / len(v), v[int(lo * (len(v) - 1))], v[int(hi * (len(v) - 1))]))
    return out
