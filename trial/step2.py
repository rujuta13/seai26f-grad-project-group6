"""step2.py: guess the performance and CPU cost of a flag set.

Uses k-nearest-neighbour regression over the input (flag) columns. For each
goal, the estimate is the median goal value of the nearest training rows, and
the range is their 10th to 90th percentile, so it reflects how much similar
flag sets disagree. Standard library only.
"""
import math

from step1 import _load, _columns

LOW_Q = 0.10
HIGH_Q = 0.90


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _as_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal column of train_csv.

    `row` is a flag set with its goal columns removed. The estimate is the
    median goal value of the nearest training rows; low..high is their
    10th..90th percentile range.
    """
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = maxi + mini
    n = len(rows)

    if n == 0:
        return {g: (0.0, -math.inf, math.inf) for g in goals}

    # Scale each input by its span so that no column dominates the distance.
    spans = {}
    for c in inputs:
        vals = [r[c] for r in rows]
        spans[c] = max(vals) - min(vals)

    x = {c: _as_float(row.get(c)) for c in inputs}

    def dist(i):
        s = 0.0
        r = rows[i]
        for c in inputs:
            span = spans[c]
            if span > 0:
                s += ((x[c] - r[c]) / span) ** 2
        return s

    d = [dist(i) for i in range(n)]
    order = sorted(range(n), key=lambda i: (d[i], i))
    k = min(n, max(3, round(math.sqrt(n))))
    near = order[:k]

    out = {}
    for g in goals:
        vals = sorted(rows[i][g] for i in near if isinstance(rows[i][g], float))
        if not vals:
            out[g] = (0.0, -math.inf, math.inf)
            continue
        out[g] = (
            _quantile(vals, 0.5),
            _quantile(vals, LOW_Q),
            _quantile(vals, HIGH_Q),
        )
    return out

