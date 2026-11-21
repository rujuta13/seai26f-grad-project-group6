"""step25.py - client question 25: blame, for campaign response and complaints.

blame(train_csv) ranks the input columns (flags/passes) by how strongly they
are tied to bad rows. The client asks two questions, each about one goal:
  - "Why do some customers never respond to campaigns?" -> RESPONSE+
    (bad = the customers who do not respond);
  - "What do the customers who complain have in common?" -> COMPLAIN-
    (bad = the customers who complain).

The two goals are ranked separately, then the two rankings are interleaved
(first column of each, then second column of each, and so on, skipping
repeats). Each question therefore gets its top columns near the front.

Bad and good groups for one goal:
- If the goal has two distinct values (such as a 0/1 flag), bad = rows at the
  worst value and good = rows at the best value.
- Otherwise bad = rows at or below the 25th percentile of the goal and good =
  rows above it at or above the 75th percentile.
- A goal with no contrast between bad and good rows contributes nothing.

Effects use step5._effect: the standardised mean difference, bad minus good.
A positive effect means the input is on, or higher, more often in bad rows.

Caveat: the ACCEPTEDCMP* input columns record earlier campaign acceptances,
which are closely related to RESPONSE+. They will rank high for the response
question and say more about past behaviour than about the cause of
non-response. Read them with that in mind.

If neither goal is present, or neither has any contrast, the ranking falls
back to step5.blame.

Reused from earlier steps:
  step1: _load
  step5: _effect, blame (fallback)
"""
from step1 import _load
from step5 import _effect
from step5 import blame as _blame_all

FOCUS = ("RESPONSE", "COMPLAIN")


def _contrast(vals):
    """Return (bad, good) row index lists for one goal's maximisation-signed
    values, or None if the goal gives no contrast."""
    n = len(vals)
    distinct = sorted(set(vals))
    if len(distinct) < 2:
        return None
    if len(distinct) == 2:
        bad = [i for i in range(n) if vals[i] == distinct[0]]
        good = [i for i in range(n) if vals[i] == distinct[-1]]
    else:
        srt = sorted(vals)
        lo = srt[int(0.25 * (n - 1))]
        hi = srt[int(0.75 * (n - 1))]
        bad = [i for i in range(n) if vals[i] <= lo]
        good = [i for i in range(n) if vals[i] >= hi and vals[i] > lo]
    if not bad or not good:
        return None
    return bad, good


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    input_cols, goal_cols, X, Y = _load(train_csv)
    n = len(X)
    d = len(input_cols)
    if n == 0 or d == 0 or not goal_cols:
        return list(input_cols)

    rankings = []
    for g, (name, _) in enumerate(goal_cols):
        if name.rstrip("+-") not in FOCUS:
            continue
        vals = [Y[i][g] for i in range(n)]
        contrast = _contrast(vals)
        if contrast is None:
            continue
        bad_idx, good_idx = contrast
        effects = [
            _effect([X[i][t] for i in bad_idx], [X[i][t] for i in good_idx])
            for t in range(d)
        ]
        rankings.append(sorted(range(d), key=lambda t: (-effects[t], t)))

    if not rankings:
        return _blame_all(train_csv)

    merged = []
    seen = set()
    for pos in range(d):
        for ranking in rankings:
            t = ranking[pos]
            if t not in seen:
                seen.add(t)
                merged.append(input_cols[t])
    return merged

