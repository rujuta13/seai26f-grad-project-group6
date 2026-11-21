"""Step 12: guess. Predict a player's goal values from his other numbers.

Thin wrapper around step2.guess. Each estimate is the median of the nearest
training rows, and low..high is their 5th..95th percentile range. For a player
unlike anything in training, step2 blends the range toward whole-training
statistics, so a wide range means low confidence. Goal columns in `row` are
ignored because only input columns affect the distance.
"""

import math

import step2


def _bound(x, default):
    """Float x, or default if x is NaN."""
    x = float(x)
    return default if math.isnan(x) else x


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a player `row` (goal columns already removed).

    Returns {goal: (estimate, low, high)}, where low..high is the range the
    true value is fairly sure to fall in. A goal with no usable training values
    gets a NaN estimate and an infinite range.
    """
    raw = step2.guess(train_csv, row)
    result = {}
    for goal, (est, low, high) in raw.items():
        est = float(est)
        if math.isnan(est):
            result[goal] = (math.nan, -math.inf, math.inf)
            continue
        low = min(_bound(low, -math.inf), est)
        high = max(_bound(high, math.inf), est)
        result[goal] = (est, low, high)
    return result

