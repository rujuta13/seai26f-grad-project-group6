"""step15.py: blame, for the two client questions.

Q1: why are some players weak from the spot (Penalties+)?
Q2: what separates the physically strong players from the rest (Strength+)?

step5.blame ranks input columns against one overall score, which can hide a
goal such as Penalties+ or Strength+ when other goals pull the mean the other
way. Here each goal is checked on its own: a row is bad on that goal when it
falls in the bottom BAD_QUANTILE of that goal's normalised values, with a
missing value counted as the worst. Each input column is blamed by its most
troublesome goal, and columns are ranked best first.

Reading, column splitting, the signed goal vector, normalisation, quantiles,
binning and the per-column blame score come from steps 1, 2, 4 and 5.
"""
from __future__ import annotations

from step1 import _columns, _read
from step2 import _quantile
from step4 import _objectives
from step5 import BAD_QUANTILE, _column_blame, _scores


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Each goal is considered separately. For each goal, the rows at or below the
    BAD_QUANTILE of that goal's normalised score are bad on that goal, and each
    input column is scored by how much more often bad rows fall in its most
    troublesome value or bin than the overall bad rate. A column's blame is the
    largest of these scores over all goals. Ties keep the file's column order.
    Every input column is returned.
    """
    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    if not inputs:
        return []
    if n == 0 or not goals:
        return list(inputs)

    obj = [_objectives(r, gmax, gmin) for r in rows]
    worst = [0.0] * len(inputs)
    for j in range(len(goals)):
        single = [[o[j]] for o in obj]
        score = _scores(single, 1)
        cut = _quantile(score, BAD_QUANTILE)
        bad = [s <= cut for s in score]
        p0 = sum(bad) / n
        for k, c in enumerate(inputs):
            b = _column_blame(rows, c, bad, p0)
            if b > worst[k]:
                worst[k] = b

    order = sorted(range(len(inputs)), key=lambda i: (-worst[i], i))
    return [inputs[i] for i in order]

