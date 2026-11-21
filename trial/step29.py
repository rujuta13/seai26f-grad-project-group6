"""Step 29: spot. Suspicion scores for customer records.

Standard library only. The cached training model is reused from step2, and the
suspicion score from step9 (input-space novelty, flag values never seen in
training, goal disagreement, goal range and bad goal cells) is reused unchanged.

This module adds one check that step9 does not make per input column:
  - range_part: a numeric input column (such as AGE or INCOME) holds a value
    outside the range seen in training for that column, in units of that range.
    An impossible age such as 230 is caught here even when the rest of the
    record looks typical.
  - A numeric input column that holds text or no value is counted like an
    unseen flag value, at the same UNSEEN_COST as step9.

"Behaves unlike anyone else" is answered by step9's input-space novelty term.
"Looks wrong" is answered by the sum of all terms.
"""

import math

from step2 import _model
from step9 import odd as _odd, UNSEEN_COST


RANGE_WEIGHT = 1.0   # points per unit of a column's training range outside that range


def _numeric_ranges(model):
    """For each input column whose training values are all numbers, return
    (lo, hi, unit). unit is the training span, or the size of lo when the span
    is zero, so that a constant column still gives a finite score."""
    ranges = {}
    rows = model["rows"]
    if not rows:
        return ranges
    for c in model["inputs"]:
        vals = [r[c] for r in rows]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        lo, hi = float(min(vals)), float(max(vals))
        span = hi - lo
        unit = span if span > 0 else max(abs(lo), 1.0)
        ranges[c] = (lo, hi, unit)
    return ranges


def _range_part(row, ranges):
    """Points for numeric inputs outside their training range, and for numeric
    inputs that are missing, text, or not finite."""
    score = 0.0
    for c, (lo, hi, unit) in ranges.items():
        v = row.get(c)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            score += UNSEEN_COST
            continue
        v = float(v)
        excess = max(lo - v, v - hi, 0.0)
        score += RANGE_WEIGHT * excess / unit
    return score


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    if not rows:
        return []
    base = _odd(train_csv, rows)
    model = _model(train_csv)
    if not model["targets"]:
        return base
    ranges = _numeric_ranges(model)
    return [float(s + _range_part(row, ranges)) for s, row in zip(base, rows)]

