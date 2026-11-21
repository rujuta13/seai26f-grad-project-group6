"""step15.py - client question 15: blame, for the two questions about strength and penalties.

blame(train_csv) ranks the input columns (flags/passes) by how strongly they
are tied to bad rows. The client's two questions concern two goals:
  - "Penalties+": why some players are weak from the spot;
  - "Strength+": what separates the physically strong players.

Goal selection: only goals whose names, without the "+"/"-" suffix, are
"Penalties" or "Strength" count toward badness. If the file has neither goal,
every goal counts, as in step5.blame.

Method (as in step5.blame, restricted to the chosen goals):
- Each chosen goal is turned into a percentile rank among the training rows
  (0 = worst, 1 = best).
- A row's badness is 1 minus its mean percentile over the chosen goals. The
  worst BAD_FRACTION of rows are "bad"; the rest are "good".
- For each input column, the effect is the standardised mean difference, bad
  minus good. A positive effect means the input is on, or higher, more often
  in bad rows.
- Columns are returned by effect, largest first. Ties keep file order.

Caveat: both questions share one ranking, so the result is the blend of the
two goals. Badness is averaged over them, so a row that is weak on one goal
but strong on the other may be treated as middling.

Reused from earlier steps:
  step1: _load
  step5: BAD_FRACTION, _percentiles, _effect
"""
from step1 import _load
from step5 import BAD_FRACTION, _effect, _percentiles

FOCUS_GOALS = ("Penalties", "Strength")


def _focus(goal_cols):
    """Positions of the goals the client asked about, or all goals if none match."""
    pos = [g for g, (name, _) in enumerate(goal_cols)
           if name.rstrip("+-") in FOCUS_GOALS]
    return pos or list(range(len(goal_cols)))


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    input_cols, goal_cols, X, Y = _load(train_csv)
    n = len(X)
    d = len(input_cols)
    if n == 0 or d == 0 or not goal_cols:
        return list(input_cols)

    use = _focus(goal_cols)
    m = len(use)

    # badness: 1 minus the mean percentile over the chosen goals (1 = worst)
    pct_cols = [_percentiles([Y[i][g] for i in range(n)]) for g in use]
    badness = [1.0 - sum(col[i] for col in pct_cols) / m for i in range(n)]

    k = max(1, int(BAD_FRACTION * n))
    order = sorted(range(n), key=lambda i: (-badness[i], i))
    bad_idx = sorted(order[:k])
    bad_set = set(bad_idx)
    good_idx = [i for i in range(n) if i not in bad_set]
    if not good_idx:
        return list(input_cols)

    effects = [
        _effect([X[i][t] for i in bad_idx], [X[i][t] for i in good_idx])
        for t in range(d)
    ]
    ranked = sorted(range(d), key=lambda t: (-effects[t], t))
    return [input_cols[t] for t in ranked]

