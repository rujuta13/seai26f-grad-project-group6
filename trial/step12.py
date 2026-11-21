"""step12.py: guess a player's goals from his other numbers, with a range
that says how confident the guess is.

guess(train_csv, row): the estimate for each goal is the median goal of the
nearest training flag sets, as in step2.guess. The range is widened so that it
is fairly sure to hold the true value. To set its width, the same
nearest-neighbour rule predicts training rows whose goal is known
(leave-one-out on a fixed sample), and the CAL_Q quantile of those absolute
errors is the half-width. The reported range is the wider of the neighbour
range and the estimate plus or minus that half-width, so a wide range means
low confidence.

Reuses step1 (loading and column roles) and step2 (quantiles and the
neighbour-count convention). Standard library only.
"""
import math
import os
import random
import statistics

from step1 import _load, _columns
from step2 import _quantile, LOW_Q, HIGH_Q

CAL_SAMPLE = 200   # training rows used for leave-one-out calibration
CAL_Q = 0.90       # target coverage of the calibrated half-width
SEED = 1           # fixed seed so results are reproducible

_CACHE = {}        # (path, mtime, size) -> fitted model for the latest file


def _as_float(v):
    """A finite float, or None if the cell is missing or not numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _k_for(n):
    """Neighbour count, as in step2.guess."""
    return min(n, max(3, round(math.sqrt(n))))


def _scale(row, inputs, stats):
    """Flag vector of a row, each input scaled to [0, 1]; missing cells use the median."""
    mins, spans, medians = stats
    x = []
    for c in inputs:
        v = _as_float(row.get(c))
        if v is None:
            v = medians[c]
        span = spans[c]
        x.append((v - mins[c]) / span if span > 0 else 0.0)
    return x


def _order(q, X, skip=None):
    """(squared distance, row index) for every row of X, nearest first."""
    d = []
    for j, x in enumerate(X):
        if j == skip:
            continue
        s = 0.0
        for a, b in zip(q, x):
            s += (a - b) ** 2
        d.append((s, j))
    d.sort()
    return d


def _near_values(ordered, yg, k):
    """Sorted goal values of the k nearest rows that have a usable goal."""
    vals = []
    for _, j in ordered:
        v = yg[j]
        if v is not None:
            vals.append(v)
            if len(vals) == k:
                break
    return sorted(vals)


def _fit(train_csv):
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = maxi + mini
    n = len(rows)

    mins, spans, medians = {}, {}, {}
    for c in inputs:
        vals = [r[c] for r in rows]
        if vals:
            mins[c] = min(vals)
            spans[c] = max(vals) - mins[c]
            medians[c] = statistics.median(vals)
        else:
            mins[c] = spans[c] = medians[c] = 0.0
    stats = (mins, spans, medians)

    X = [_scale(r, inputs, stats) for r in rows]
    Y = {g: [_as_float(r.get(g)) for r in rows] for g in goals}
    cnt = {g: sum(v is not None for v in Y[g]) for g in goals}

    # Leave-one-out calibration: how far the nearest-neighbour median misses.
    resid = {g: [] for g in goals}
    if n > 1:
        rng = random.Random(SEED)
        for i in rng.sample(range(n), min(n, CAL_SAMPLE)):
            ordered = _order(X[i], X, skip=i)
            for g in goals:
                y = Y[g][i]
                if y is None:
                    continue
                vals = _near_values(ordered, Y[g], _k_for(cnt[g]))
                if vals:
                    resid[g].append(abs(y - statistics.median(vals)))

    half = {}
    for g in goals:
        r = sorted(resid[g])
        half[g] = _quantile(r, CAL_Q) if r else math.inf

    return {"goals": goals, "inputs": inputs, "stats": stats, "X": X,
            "Y": Y, "cnt": cnt, "half": half}


def _model(train_csv):
    key = (os.path.abspath(train_csv),
           os.path.getmtime(train_csv),
           os.path.getsize(train_csv))
    m = _CACHE.get(key)
    if m is None:
        m = _fit(train_csv)
        _CACHE.clear()
        _CACHE[key] = m
    return m


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal column of train_csv.

    `row` is a flag set with its goal columns removed. The estimate is the
    median goal of the nearest training flag sets. low..high is fairly sure to
    hold the true value: the wider of the neighbours' 10th..90th percentile
    range and the estimate +/- the calibrated leave-one-out error.
    """
    m = _model(train_csv)
    goals = m["goals"]
    unknown = (0.0, -math.inf, math.inf)
    if not m["X"]:
        return {g: unknown for g in goals}

    ordered = _order(_scale(row, m["inputs"], m["stats"]), m["X"])
    out = {}
    for g in goals:
        if m["cnt"][g] == 0:
            out[g] = unknown
            continue
        vals = _near_values(ordered, m["Y"][g], _k_for(m["cnt"][g]))
        est = _quantile(vals, 0.5)
        lo = _quantile(vals, LOW_Q)
        hi = _quantile(vals, HIGH_Q)
        w = m["half"][g]
        out[g] = (est, min(lo, est - w), max(hi, est + w))
    return out

