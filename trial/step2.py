"""Step 2: guess how fast a flag set will run and how much CPU it costs.

The guess comes from the training rows whose flag settings are closest to
the given row. For each goal, the estimate is the median of those
neighbours' values, and the range is their 10th to 90th percentile.
"""
import math

from step1 import _load


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _distance(row, other, cols):
    """Sum of absolute differences over the input columns both rows share."""
    d = 0.0
    for c in cols:
        a = row[c]
        b = other.get(c)
        if isinstance(a, float) and isinstance(b, float):
            d += abs(a - b)
        elif a != b:
            d += 1.0
    return d


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal in the training file.

    `row` is a dict of input columns only (goal columns removed). The
    estimate is the median goal value among the nearest training rows, and
    low..high is their 10th..90th percentile range.
    """
    header, rows = _load(train_csv)
    goals = [c for c in header if _is_goal(c)]
    if not goals:
        return {}

    n = len(rows)
    if n == 0:
        return {g: (0.0, 0.0, 0.0) for g in goals}

    # Input columns that appear both in the training file and in the query row.
    input_cols = [
        c for c in header
        if not _is_goal(c) and not c.endswith("X") and c in row
    ]

    dists = [_distance(row, r, input_cols) for r in rows]
    order = sorted(range(n), key=lambda i: (dists[i], i))
    k = min(n, max(10, int(math.sqrt(n))))
    nearest = order[:k]

    out = {}
    for g in goals:
        vals = sorted(
            rows[i][g] for i in nearest if isinstance(rows[i].get(g), float)
        )
        if len(vals) < 3:
            # Too few usable neighbours: fall back to all training values.
            vals = sorted(
                r[g] for r in rows if isinstance(r.get(g), float)
            )
        if not vals:
            out[g] = (0.0, 0.0, 0.0)
            continue

        est = _quantile(vals, 0.5)
        low = min(_quantile(vals, 0.1), est)
        high = max(_quantile(vals, 0.9), est)
        out[g] = (float(est), float(low), float(high))

    return out

