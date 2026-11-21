"""step25.py: blame, for the two client questions.

Q1: Why do some customers never respond to campaigns? (RESPONSE+)
Q2: What do the customers who complain have in common? (COMPLAIN-)

step15.blame checks every goal with a bottom-quarter cut. That cut fails for
goals where most rows tie at the best value, such as COMPLAIN- (about 1% of
customers complain), because the cut lands on the best value and every row
counts as bad. This module keeps step15's approach but changes the bad-row rule:
a row is bad on a goal when its normalised score is at or below the bottom-quarter
cut, unless that cut sits at the best value. In that case, a row is bad when it
falls strictly below the best value. The client goals (RESPONSE and COMPLAIN) are
used when present, and every goal otherwise.

Reading, column splitting, the signed goal vector, normalisation, quantiles and
the per-column blame score come from steps 1, 2, 4 and 5.
"""
from __future__ import annotations

from step1 import _columns, _read
from step2 import _quantile
from step4 import _objectives
from step5 import BAD_QUANTILE, _column_blame, _scores

CLIENT_GOALS = ("RESPONSE", "COMPLAIN")   # base names of the goals the client asks about


def _is_client_goal(g: str) -> bool:
    """True if goal column g (e.g. 'RESPONSE+') is one of the client's goals."""
    return g[:-1] in CLIENT_GOALS


def _bad_rows(score: list[float]) -> list[bool]:
    """Bad flags for one goal's normalised scores in [0, 1], higher is better.

    Normally bad means at or below the BAD_QUANTILE cut. When the cut sits at the
    best value (most rows tie there, as with a rare complaint), bad means strictly
    below the best value, so the rare worst rows are still flagged.
    """
    cut = _quantile(score, BAD_QUANTILE)
    top = max(score)
    if cut < top:
        return [s <= cut for s in score]
    return [s < top for s in score]


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Each client goal (RESPONSE+ and COMPLAIN-, when present) is considered
    separately. Each goal's bad rows come from _bad_rows, and each input column
    is scored by how much more often bad rows fall in its most troublesome value
    or bin than the overall bad rate. A column's blame is the largest such score
    over the goals considered. If neither client goal is present, every goal is
    used. Ties keep the file's column order. Every input column is returned.
    """
    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    if not inputs:
        return []
    if n == 0 or not goals:
        return list(inputs)

    focus = [g for g in goals if _is_client_goal(g)] or goals
    obj = [_objectives(r, gmax, gmin) for r in rows]

    worst = [0.0] * len(inputs)
    for g in focus:
        j = goals.index(g)
        single = [[o[j]] for o in obj]
        score = _scores(single, 1)
        bad = _bad_rows(score)
        p0 = sum(bad) / n
        for k, c in enumerate(inputs):
            b = _column_blame(rows, c, bad, p0)
            if b > worst[k]:
                worst[k] = b

    order = sorted(range(len(inputs)), key=lambda i: (-worst[i], i))
    return [inputs[i] for i in order]

