"""Step 2: guess each goal of a flag set before benchmarking it.

Each guess is the median goal value of the k most similar training rows
(similarity over the input/flag columns only). The range is the 10th to
90th percentile of those neighbours. When the neighbourhood is far from
the query, the range is widened to the 5th to 95th percentile of the
whole training set, so the reported uncertainty grows with distance.
"""

import math
import os

from step1 import _load, _split_columns

K = 15
FAR_THRESHOLD = 0.25

_CACHE = {}


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _load_cached(train_csv):
    key = (train_csv, os.path.getmtime(train_csv))
    if key not in _CACHE:
        _CACHE[key] = _load(train_csv)
    return _CACHE[key]


def _as_float(value):
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _spec(rows, inputs):
    """Per input column: ('num', lo, hi) if every training value is numeric, else ('cat', None, None)."""
    spec = {}
    for c in inputs:
        vals = [r.get(c) for r in rows]
        if vals and all(isinstance(v, float) for v in vals):
            spec[c] = ("num", min(vals), max(vals))
        else:
            spec[c] = ("cat", None, None)
    return spec


def _distance(train_row, row, inputs, spec):
    """Sum of per-column differences; numeric columns are scaled to [0, 1]."""
    d = 0.0
    for c in inputs:
        kind, lo, hi = spec[c]
        a = train_row.get(c)
        b = row.get(c)
        if kind == "num":
            fa = _as_float(a)
            fb = _as_float(b)
            if fa is None or fb is None:
                d += 1.0
            else:
                span = hi - lo
                d += abs(fa - fb) / span if span > 0 else 0.0
        else:
            d += 0.0 if str(a) == str(b) else 1.0
    return d


def guess(train_csv, row):
    """Return {goal: (estimate, low, high)} for every goal column of the training file.

    `row` maps input column names to values; goal columns are absent.
    low..high is the range the true value is expected to fall in.
    """
    columns, rows = _load_cached(train_csv)
    maximise, minimise, inputs = _split_columns(columns)
    goals = maximise + minimise
    n = len(rows)

    if n == 0 or not goals:
        return {g: (0.0, -math.inf, math.inf) for g in goals}

    spec = _spec(rows, inputs)
    ranked = sorted(
        ((_distance(rows[i], row, inputs, spec), i) for i in range(n)),
        key=lambda t: (t[0], t[1]),
    )
    k = min(K, n)
    near = ranked[:k]
    neighbours = [i for _, i in near]

    mean_d = sum(d for d, _ in near) / k
    width = max(1, len(inputs))
    far = (mean_d / width) > FAR_THRESHOLD

    out = {}
    for g in goals:
        local = sorted(
            v for v in (_as_float(rows[i].get(g)) for i in neighbours) if v is not None
        )
        everything = sorted(
            v for v in (_as_float(r.get(g)) for r in rows) if v is not None
        )

        if not local:
            local = everything
        if not everything:
            out[g] = (0.0, -math.inf, math.inf)
            continue

        estimate = _quantile(local, 0.5)
        if far:
            low = _quantile(everything, 0.05)
            high = _quantile(everything, 0.95)
        else:
            low = _quantile(local, 0.10)
            high = _quantile(local, 0.90)

        estimate = min(max(estimate, low), high)
        out[g] = (float(estimate), float(low), float(high))

    return out

