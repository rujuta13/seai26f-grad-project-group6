"""Step 25: blame for the two client questions.

Standard library only. Loading, column splitting, goal normalisation and the
per-column blame are reused from steps 1 and 15. The bad-share setting is
reused from step 5.

Client questions:
  - "Why do some customers never respond to campaigns?" -> the RESPONSE+ goal.
  - "What do the customers who complain have in common?" -> the COMPLAIN- goal.

Method: blame() only ranks input columns, so it is restricted to the goals
named RESPONSE+ and COMPLAIN- when the file has them, and falls back to every
goal otherwise, as step 15 does. For each chosen goal, the bad rows are the
worst BAD_SHARE of rows on that goal. These goals are usually binary, so many
rows tie. Step 15 would cut such a tie arbitrarily by row index. Here whole
groups of equal value are taken from the worst end while they fit within the
bad share, and the worst group is always taken. A goal that is 0 for most
customers ("never responds") is therefore judged as a whole group. Each
column's blame is step 15's per-goal blame, summed over the chosen goals.
"""

from step1 import _load, _columns, _normalised_goals
from step5 import BAD_SHARE
from step15 import _column_blame


CLIENT_GOALS = ("RESPONSE+", "COMPLAIN-")


def _bad_rows_by_value(values, n_bad):
    """Indices of the worst rows on one goal. Whole groups of equal normalised
    value are taken from the worst (lowest) end while they fit within n_bad.
    The worst group is always taken, even if it is larger than n_bad, so a
    tie is never split by row index. Continuous goals give exactly n_bad rows."""
    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(v, []).append(i)
    bad = set()
    for v in sorted(groups):
        members = groups[v]
        if bad and len(bad) + len(members) > n_bad:
            break
        bad.update(members)
    return bad


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Bad rows are judged on RESPONSE+ and COMPLAIN- when the file has them,
    otherwise on every goal."""
    header, rows = _load(train_csv)
    n = len(rows)
    inputs, goals = _columns(header)
    if n == 0 or not inputs or not goals:
        return list(inputs)

    chosen = [goal for goal in goals if goal[0] in CLIENT_GOALS] or list(goals)
    n_bad_target = max(1, int(n * BAD_SHARE))

    totals = dict.fromkeys(inputs, 0.0)
    for goal in chosen:
        vals = [p[0] for p in _normalised_goals(rows, [goal])]
        bad = _bad_rows_by_value(vals, n_bad_target)
        base = len(bad) / n
        for c in inputs:
            totals[c] += _column_blame(rows, c, bad, n, base)

    order = sorted(range(len(inputs)), key=lambda k: (-totals[inputs[k]], k))
    return [inputs[k] for k in order]

