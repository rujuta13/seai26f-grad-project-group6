"""step22.py: guess, for the two client questions.

Q1: "Guess how much a new customer will buy online versus in store."
Q2: "Guess whether this customer will answer our next campaign."

Both answers come from step12.guess, which reports every goal column with an
estimate and a range. The purchase goals (NUMWEBPURCHASES+, NUMSTOREPURCHASES+)
and the campaign response goal (RESPONSE+) are all included, so the caller reads
the ones it needs. A goal with no usable training values gets NaN and an
unbounded range, meaning no confident guess.

Reading, column splitting and the neighbour guess come from steps 1, 2 and 12.
Only the wrapper that fixes the return shape is new here.
"""
from __future__ import annotations

from step12 import guess as _guess


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a customer described by row (goal columns removed).

    Returns {goal: (estimate, low, high)} for every goal column in train_csv.
    low..high is the range the true value is expected to fall in. A wider range
    means less confidence. A goal with no usable training values gets
    (NaN, -inf, inf): no estimate and no confidence.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for goal, (est, low, high) in _guess(train_csv, row).items():
        out[goal] = (float(est), float(low), float(high))
    return out

