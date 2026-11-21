"""step25.py: which input flags are to blame for customers who never respond to
campaigns, and what customers who complain have in common?

Blame is judged goal by goal, as in step15, so that each client question is
answered by its own goal (RESPONSE+ for non-response, COMPLAIN- for complaints).
On each goal, the bad rows are the worst values of that goal, taken whole by
value level and capped at BEST_FRACTION of the rows. Step15's cut-off rule keeps
every row tied at the cut-off, which for a rare 0/1 goal such as COMPLAIN- would
mark almost every customer as bad. Here a tied level is taken whole only while
the total stays within the cap, or when it is the worst level and so must be
taken alone. Each input is scored on each goal with step5's scoring rule, and
its blame is the highest of those scores. Inputs are returned most to blame
first, with ties kept in file order. Standard library only.
"""
import math

from step1 import _load, _columns, BEST_FRACTION
from step5 import _score


def _bad_on(goal_rows, col, sgn):
    """Indices into goal_rows of the worst rows on one goal.

    sgn is +1 for a maximise goal, -1 for a minimise goal. Rows are grouped by
    their badness (higher means worse). Whole groups are taken from the worst
    down while the total stays within the cap. The worst group is always taken,
    even when it alone exceeds the cap, so a tied majority is not split.
    """
    n = len(goal_rows)
    cap = max(1, int(BEST_FRACTION * n))
    levels = {}
    for i, r in enumerate(goal_rows):
        levels.setdefault(-sgn * r[col], []).append(i)

    bad = set()
    for b in sorted(levels, reverse=True):
        members = levels[b]
        if bad and len(bad) + len(members) > cap:
            break
        bad.update(members)
        if len(bad) >= cap:
            break
    return bad


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
                 if all(isinstance(r[c], float) and math.isfinite(r[c])
                        for c, _ in goals)]
    if not goals or not goal_rows:
        return list(inputs)

    bad_sets = [_bad_on(goal_rows, c, sgn) for c, sgn in goals]
    scores = {c: max(_score(c, goal_rows, bad) for bad in bad_sets)
              for c in inputs}
    order = sorted(range(len(inputs)),
                   key=lambda k: (-scores[inputs[k]], k))
    return [inputs[k] for k in order]

