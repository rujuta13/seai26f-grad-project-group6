"""step5.py: which input flags are to blame for bad rows?

Bad rows are the worst rows on all goals together: the anti-Pareto front (rows
that no other row is worse than on every goal and worse on at least one),
capped at BEST_FRACTION of the rows, as step1.best does for good rows. Each
input is scored by how differently its values are spread among bad rows than
among all rows:
  - a column with few distinct values: the largest gap between the share of
    bad rows and the share of all rows holding that value;
  - a numeric column with many distinct values: the gap between the mean among
    bad rows and the mean among all rows, scaled by the column's span.
Inputs are returned most to blame first. Standard library only.
"""
from step1 import _load, _columns, _dominates, BEST_FRACTION

CATEGORY_LIMIT = 10  # up to this many distinct values, compare shares


def _bad_front(goal_rows, goals):
    """Indices (into goal_rows) of the worst rows on all goals together."""
    n = len(goal_rows)
    # Orient so that higher means worse; the Pareto front is then the worst rows.
    B = [[-sgn * float(r[c]) for c, sgn in goals] for r in goal_rows]
    front = [i for i in range(n)
             if not any(_dominates(B[j], B[i]) for j in range(n) if j != i)]
    cap = max(1, int(BEST_FRACTION * n))
    if len(front) > cap:
        g = len(goals)
        lo = [min(B[i][t] for i in range(n)) for t in range(g)]
        hi = [max(B[i][t] for i in range(n)) for t in range(g)]

        def badness(i):
            s = 0.0
            for t in range(g):
                span = hi[t] - lo[t]
                s += (B[i][t] - lo[t]) / span if span > 0 else 1.0
            return s

        front.sort(key=lambda i: (-badness(i), i))
        front = front[:cap]
    return front


def _share_gap(col, goal_rows, bad):
    """Largest gap between the share of bad rows and of all rows holding a value."""
    n = len(goal_rows)
    nb = len(bad)
    count_all, count_bad = {}, {}
    for i, r in enumerate(goal_rows):
        v = r[col]
        count_all[v] = count_all.get(v, 0) + 1
        if i in bad:
            count_bad[v] = count_bad.get(v, 0) + 1
    return max(abs(count_bad.get(v, 0) / nb - count_all[v] / n)
               for v in count_all)


def _mean_gap(col, goal_rows, bad):
    """Gap between the mean among bad rows and among all rows, over the span."""
    vals = [r[col] for r in goal_rows]
    span = max(vals) - min(vals)
    if span <= 0:
        return 0.0
    all_mean = sum(vals) / len(vals)
    bad_vals = [vals[i] for i in bad]
    bad_mean = sum(bad_vals) / len(bad_vals)
    return abs(bad_mean - all_mean) / span


def _score(col, goal_rows, bad):
    distinct = {r[col] for r in goal_rows}
    if (all(isinstance(v, float) for v in distinct)
            and len(distinct) > CATEGORY_LIMIT):
        return _mean_gap(col, goal_rows, bad)
    return _share_gap(col, goal_rows, bad)


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    An input scores high when its values are unusually concentrated among the
    worst rows on all goals together. Inputs with equal scores keep file order.
    """
    cols, rows = _load(train_csv)
    _, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    goal_names = {c for c, _ in goals}
    inputs = [c for c in cols
              if not c.endswith("X") and c not in goal_names]
    if not inputs:
        return []

    goal_rows = [r for r in rows
                 if all(isinstance(r[c], float) for c, _ in goals)]
    if not goals or not goal_rows:
        return list(inputs)

    bad = set(_bad_front(goal_rows, goals))
    scores = {c: _score(c, goal_rows, bad) for c in inputs}
    order = sorted(range(len(inputs)),
                   key=lambda k: (-scores[inputs[k]], k))
    return [inputs[k] for k in order]

