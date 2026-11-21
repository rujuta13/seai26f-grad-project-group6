"""step11.py: player types (families) and the strong-and-good-from-the-spot front.

families(train_csv): step1's k-means family labels, reused unchanged.

best(train_csv): the same rule as step1.best (the Pareto front of the goals,
capped at BEST_FRACTION of the rows, ranked by normalised goal sum). step1
compares every row with every other row, which is O(n^2) dominance checks and
too slow for about 17,000 players. Here rows are sorted by their oriented goal
vector, best first. A dominator always sorts before the row it dominates, so
each row is checked only against the front found so far. The result is the same
as step1.best.

Standard library only.
"""
import step1 as _step1
from step1 import _load, _columns, _dominates, BEST_FRACTION


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return _step1.families(train_csv)


def _pareto_front(V):
    """Indices of rows in V that no other row dominates (higher is better)."""
    order = sorted(range(len(V)), key=lambda i: V[i], reverse=True)
    front = []
    for i in order:
        if not any(_dominates(V[j], V[i]) for j in front):
            front.append(i)
    return front


def _top_by_score(V, front, cap):
    """The cap front rows with the highest normalised goal sum (as in step1.best)."""
    g = len(V[0])
    lo = [min(v[t] for v in V) for t in range(g)]
    hi = [max(v[t] for v in V) for t in range(g)]

    def score(i):
        s = 0.0
        for t in range(g):
            span = hi[t] - lo[t]
            s += (V[i][t] - lo[t]) / span if span > 0 else 1.0
        return s

    scores = {i: score(i) for i in front}
    ranked = sorted(front, key=lambda i: (-scores[i], i))
    return ranked[:cap]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows."""
    cols, rows = _load(train_csv)
    n = len(rows)
    cap = int(BEST_FRACTION * n)
    if n == 0 or cap == 0:
        return []
    _, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals:
        return []

    # Orient every goal so that higher is better.
    V = [[sgn * float(r[c]) for c, sgn in goals] for r in rows]

    front = _pareto_front(V)
    if len(front) > cap:
        front = _top_by_score(V, front, cap)
    return sorted(front)

