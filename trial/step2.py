"""step2.py - client question 3: guess the goals of a flag set.

guess(train_csv, row) estimates every goal of an unseen flag set from the
training rows whose flag sets are most similar to it (k nearest neighbours
in the standardised flag space). The estimate is the median of the
neighbours' goal values. low..high is the 10th..90th percentile range of
those neighbours, so the range widens when similar flag sets disagree.

Goal keys are the goal column names as they appear in the csv, suffix
included (e.g. "Performance-", "Cpu-"). Values are in original units.

Loading and distance helpers come from step1.
"""
import functools
import math
import statistics

from step1 import _load, _sqdist

K_NEIGHBOURS = 10
LOW_Q = 0.10
HIGH_Q = 0.90


@functools.lru_cache(maxsize=None)
def _model(train_csv):
    """Load the training data once per path and precompute what guess needs."""
    input_cols, goal_cols, X, Y = _load(train_csv)
    n = len(X)
    d = len(input_cols)

    means = [0.0] * d
    stds = [1.0] * d
    if n:
        for t in range(d):
            means[t] = sum(x[t] for x in X) / n
            var = sum((x[t] - means[t]) ** 2 for x in X) / n
            s = math.sqrt(var)
            stds[t] = s if s > 0 else 1.0

    Z = [[(x[t] - means[t]) / stds[t] for t in range(d)] for x in X]

    # _load negates minimised goals; undo that to get original units.
    goal_vals = [
        [sign * Y[i][g] for g, (_, sign) in enumerate(goal_cols)]
        for i in range(n)
    ]
    return input_cols, goal_cols, means, stds, Z, goal_vals


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def guess(train_csv, row):
    """Estimate every goal of a flag set.

    `row` maps input column names to values (goal columns removed).
    Returns {goal: (estimate, low, high)} for every goal column.
    """
    input_cols, goal_cols, means, stds, Z, goal_vals = _model(train_csv)
    n = len(Z)
    if n == 0:
        raise ValueError("training file has no data rows")

    missing = [c for c in input_cols if c not in row]
    if missing:
        raise KeyError(f"row is missing input columns: {missing}")

    z = [(float(row[c]) - means[t]) / stds[t] for t, c in enumerate(input_cols)]

    # k nearest training rows; ties broken by row index for determinism
    order = sorted(range(n), key=lambda i: (_sqdist(Z[i], z), i))
    nbrs = order[:min(K_NEIGHBOURS, n)]

    result = {}
    for g, (name, _) in enumerate(goal_cols):
        vals = sorted(goal_vals[i][g] for i in nbrs)
        est = statistics.median(vals)
        low = min(_quantile(vals, LOW_Q), est)
        high = max(_quantile(vals, HIGH_Q), est)
        result[name] = (float(est), float(low), float(high))
    return result

