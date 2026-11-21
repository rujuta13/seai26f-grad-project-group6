"""step15.py: which input flags are to blame for weak players on each goal?

The client asks why some players are weak from the spot (Penalties+) and what
separates physically strong players (Strength+) from the rest. Each is a single
goal, so blame is judged goal by goal rather than on the anti-Pareto front used
in step5. On each goal, a row is bad when it is among the worst BEST_FRACTION of
rows on that goal alone, with ties at the cut-off kept. Each input is scored on
each goal with step5's scoring rule, and its blame is the highest of those
scores. Inputs are returned most to blame first, with ties kept in file order.
Standard library only.
"""
from step1 import _load, _columns, BEST_FRACTION
from step5 import _score


def _worst_on(goal_rows, col, sgn):
    """Indices into goal_rows of the worst BEST_FRACTION of rows on one goal
    (ties at the cut-off are kept). sgn is +1 for a maximise goal, -1 for minimise."""
    n = len(goal_rows)
    cap = max(1, int(BEST_FRACTION * n))
    # Higher badness means worse on this goal.
    badness = [-sgn * r[col] for r in goal_rows]
    cutoff = sorted(badness, reverse=True)[cap - 1]
    return {i for i, b in enumerate(badness) if b >= cutoff}


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    An input is blamed when its values are unusually concentrated among the
    worst rows on some goal. Its score is the highest score over the goals.
    Inputs with equal scores keep file order.
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

    bad_sets = [_worst_on(goal_rows, c, sgn) for c, sgn in goals]
    scores = {c: max(_score(c, goal_rows, bad) for bad in bad_sets)
              for c in inputs}
    order = sorted(range(len(inputs)),
                   key=lambda k: (-scores[inputs[k]], k))
    return [inputs[k] for k in order]

