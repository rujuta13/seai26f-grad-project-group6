"""Step 2: guess how fast a flag set is and how much CPU it costs, before benchmarking.

Method: k-nearest-neighbour regression over the training rows. The flag sets
closest to the query (mean per-column mismatch over the input columns) are
found, and each goal's estimate is the median of those neighbours' measured
values.

The low..high range is the 5th..95th percentile of the neighbours' values.
When the query is unlike anything in training (the nearest neighbours are far
away), the estimate and range are blended toward the whole-training-set
statistics, because the local neighbourhood says little about such a query.
"""

import math
from functools import lru_cache

from step1 import _read, _classify

_K = 15            # neighbours used for the local estimate
_MIN_LOCAL = 5     # fewer usable neighbours than this -> rely on global statistics
_LOW_Q = 0.05      # low end of the range (quantile of neighbour values)
_HIGH_Q = 0.95     # high end of the range
_FAR = 0.5         # mean neighbour distance at which the local estimate is fully distrusted

_NAN = float("nan")
_INF = math.inf


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


@lru_cache(maxsize=4)
def _load(train_csv):
    """Parse train_csv once. Returns (cols, rows, goal_values, goal_names).

    cols: [(name, kind, lo, span)] for each input column; kind is "num" or "cat".
    rows: the training rows (dicts).
    goal_values: {goal: {row index: value}} over valid (numeric, non-NaN) cells.
    goal_names: goal column names exactly as they appear in the header.
    """
    header, rows = _read(train_csv)
    inputs, goals_max, goals_min = _classify(header)

    cols = []
    for name in inputs:
        values = [r.get(name) for r in rows]
        if values and all(isinstance(v, float) for v in values):
            finite = [v for v in values if not math.isnan(v)]
            if finite:
                lo, hi = min(finite), max(finite)
                cols.append((name, "num", lo, hi - lo))
                continue
        cols.append((name, "cat", 0.0, 0.0))

    goal_names = goals_max + goals_min
    goal_values = {
        g: {i: r[g] for i, r in enumerate(rows) if _is_num(r.get(g))}
        for g in goal_names
    }
    return cols, rows, goal_values, goal_names


def _distance(cols, train_row, query):
    """Mean per-column mismatch in [0, 1] between a training row and the query."""
    if not cols:
        return 0.0
    total = 0.0
    for name, kind, lo, span in cols:
        a = train_row.get(name)
        b = query.get(name)
        if kind == "num":
            if _is_num(a) and _is_num(b):
                total += min(1.0, abs(a - b) / span) if span > 0 else 0.0
            else:
                total += 0.5
        else:
            if a is None or b is None:
                total += 0.5
            elif a != b:
                total += 1.0
    return total / len(cols)


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a flag set `row` (goal columns already removed).

    Returns {goal: (estimate, low, high)}, where low..high is the range the
    true value is expected to fall in with high confidence.
    """
    cols, rows, goal_values, goal_names = _load(train_csv)
    if not rows:
        return {g: (_NAN, -_INF, _INF) for g in goal_names}

    ranked = sorted((_distance(cols, tr, row), i) for i, tr in enumerate(rows))

    # Blend weight from the nearest neighbours: close neighbours mean the local
    # estimate is trusted (far ~ 0); distant neighbours push toward global stats.
    near = ranked[:_K]
    mean_d = sum(d for d, _ in near) / len(near)
    far = min(1.0, mean_d / _FAR)

    result = {}
    for g in goal_names:
        vals = goal_values[g]
        if not vals:
            result[g] = (_NAN, -_INF, _INF)
            continue

        global_sorted = sorted(vals.values())

        # Up to _K nearest rows that actually have a value for this goal.
        local = []
        for _, i in ranked:
            v = vals.get(i)
            if v is not None:
                local.append(v)
                if len(local) >= _K:
                    break
        local_sorted = sorted(local)

        w = far if len(local_sorted) >= _MIN_LOCAL else 1.0

        est = ((1 - w) * _quantile(local_sorted, 0.5)
               + w * _quantile(global_sorted, 0.5))
        low = ((1 - w) * _quantile(local_sorted, _LOW_Q)
               + w * _quantile(global_sorted, _LOW_Q))
        high = ((1 - w) * _quantile(local_sorted, _HIGH_Q)
                + w * _quantile(global_sorted, _HIGH_Q))

        low = min(low, est)
        high = max(high, est)
        result[g] = (float(est), float(low), float(high))
    return result

