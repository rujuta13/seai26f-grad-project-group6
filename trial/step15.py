"""Step 15: blame, per goal. Why are some players weak from the spot, and
what separates the physically strong players from the rest?

Standard library only. Loading, column splitting and goal normalisation are
reused from step1. The bad-share and minimum-group settings are reused from
step5.

Method: step 5 pools all goals into one score, so a player who is weak on
penalties but strong on strength can average out to 'fine' and hide the
problem. Here each goal is judged on its own. For every goal, a row is bad on
that goal when its normalised value is in the worst BAD_SHARE of the training
rows. For each goal and each input column, the blame is computed as in step 5,
using that goal's bad rows. A column's blame is the sum of its blame over all
goals, so a column that explains weak penalties or weak strength ranks high.
"""

from step1 import _load, _columns, _normalised_goals
from step5 import BAD_SHARE, MIN_GROUP


def _bad_rows(values, n_bad):
    """Indices of the n_bad lowest values, ties broken by row index."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    return set(order[:n_bad])


def _column_blame(rows, col, bad, n, base):
    """Step 5's blame for one input column, given the set of bad row indices."""
    groups = {}
    for i, row in enumerate(rows):
        g = groups.setdefault(row[col], [0, 0])
        g[0] += 1
        if i in bad:
            g[1] += 1

    total = 0.0
    for count, bad_count in groups.values():
        if count < MIN_GROUP:
            continue
        lift = bad_count / count - base
        if lift > 0:
            total += lift * count / n
    return total


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    header, rows = _load(train_csv)
    n = len(rows)
    inputs, goals = _columns(header)
    if n == 0 or not inputs or not goals:
        return list(inputs)

    n_bad = max(1, int(n * BAD_SHARE))
    base = n_bad / n

    totals = dict.fromkeys(inputs, 0.0)
    for goal in goals:
        vals = [p[0] for p in _normalised_goals(rows, [goal])]
        bad = _bad_rows(vals, n_bad)
        for c in inputs:
            totals[c] += _column_blame(rows, c, bad, n, base)

    order = sorted(range(len(inputs)), key=lambda k: (-totals[inputs[k]], k))
    return [inputs[k] for k in order]

