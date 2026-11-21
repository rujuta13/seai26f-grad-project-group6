"""step25.py: blame the input columns for customers who ignore campaigns or complain.

Standard library only (Python 3.13). Reuses step1 (csv helpers, goal
classification and normalisation) and step5 (the grouping keys and the
adjusted eta-squared that step5 uses to rank input columns).

The client asks about two outcomes: RESPONSE+ (answers the campaign, higher
is better) and COMPLAIN- (complains, lower is better). step5.blame ranks
input columns by badness over every goal, which mixes in spend and purchase
goals that the client did not ask about. This step ranks by badness over
only these two goals. A row's badness is 1 minus the mean of its normalised
RESPONSE+ and COMPLAIN- scores (step1 scaling, 1 is best), so a customer who
never responds or who complains is bad. If either column is missing, blame
falls back to step5.blame.
"""
from __future__ import annotations

import os

from step1 import _classify, _goal_scores, _read
from step5 import _adjusted_eta_squared, _keys
from step5 import blame as _step5_blame

_TARGETS = ("RESPONSE+", "COMPLAIN-")


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    A column ranks higher the more its values separate bad customers (never
    responding to campaigns, or complaining) from good ones. Columns with no
    explanatory power keep their header order at the end.
    """
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    names = [header[j] for j in inputs]
    if not rows or not inputs or not goals:
        return names

    wanted = [(j, sign) for j, sign in goals if header[j] in _TARGETS]
    if len(wanted) != len(_TARGETS):
        return _step5_blame(key)

    S = _goal_scores(rows, wanted)
    bad = [1.0 - sum(s) / len(s) for s in S]

    scored: list[tuple[float, int]] = []
    for pos, j in enumerate(inputs):
        keys = _keys(rows, j)
        scored.append((-_adjusted_eta_squared(keys, bad), pos))
    scored.sort()
    return [names[pos] for _, pos in scored]

