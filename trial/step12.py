"""step12.py: guess the goals of an unbenchmarked row, with calibrated confidence.

Standard library only (Python 3.13). Reuses step2 (training model, the kNN
point estimate guess() and _K, _model, _quantile) and nothing else.

step2.guess gives each goal's median over the nearest training rows and a
10th..90th percentile band. That band only reflects how much those rows
disagree with each other, which is often too narrow: a build can be unlike
its neighbours on a goal without the neighbours disagreeing at all.

This step widens the band using the training file itself. For a sample of
training rows, each goal is predicted from the other rows (leave one out)
and the absolute errors are recorded. Their 80th percentile is a half-width
that should hold for about 4 in 5 unseen rows. The reported band is the
estimate plus or minus the larger of that half-width and the neighbours' own
spread, so low..high is meant to be a range the true value falls in about
80% of the time. When a goal has no usable neighbour values, the estimate
and both bounds are nan.
"""
from __future__ import annotations

import math
import os
import random

from step2 import _K, _model, _quantile
from step2 import guess as _neighbour_guess

_COVER = 0.80       # share of unseen rows the band should hold
_CAL_SAMPLE = 300   # training rows used for the leave-one-out errors
_SEED = 1

_cal_cache: dict[str, dict[str, float]] = {}


def _knn_median(order: list[tuple[float, int]], values: list[float | None],
                skip: int) -> float | None:
    """Median of the goal values of the _K nearest rows, skipping row `skip`."""
    picked: list[float] = []
    for _, i in order:
        if i == skip:
            continue
        v = values[i]
        if v is not None:
            picked.append(v)
            if len(picked) == _K:
                break
    if not picked:
        return None
    return _quantile(sorted(picked), 0.5)


def _widths(key: str) -> dict[str, float]:
    """Calibrated half-width per goal, from leave-one-out errors on training."""
    cached = _cal_cache.get(key)
    if cached is not None:
        return cached

    model = _model(key)
    Z = model["Z"]
    goal_vals = model["goal_vals"]
    names = model["goal_names"]
    n = len(Z)

    errors: list[list[float]] = [[] for _ in names]
    if n >= 2:
        rng = random.Random(_SEED)
        idx = sorted(rng.sample(range(n), min(n, _CAL_SAMPLE)))
        for i in idx:
            zi = Z[i]
            order = sorted((math.dist(zi, z), j) for j, z in enumerate(Z))
            for g, values in enumerate(goal_vals):
                truth = values[i]
                if truth is None:
                    continue
                pred = _knn_median(order, values, i)
                if pred is not None:
                    errors[g].append(abs(pred - truth))

    widths: dict[str, float] = {}
    for g, name in enumerate(names):
        errs = sorted(errors[g])
        widths[name] = _quantile(errs, _COVER) if errs else math.nan

    _cal_cache[key] = widths
    return widths


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of an unbenchmarked row, with a confidence band.

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns {goal: (estimate, low, high)}. The estimate is the kNN median from
    step2.guess. low..high is meant to hold the true value about 80% of the
    time: it is the estimate plus or minus the larger of the neighbours'
    spread and the leave-one-out error of the training file. A goal with no
    usable neighbour values gets (nan, nan, nan).
    """
    key = os.fspath(train_csv)
    widths = _widths(key)
    base = _neighbour_guess(key, row)

    out: dict[str, tuple[float, float, float]] = {}
    for name, (est, lo, hi) in base.items():
        if math.isnan(est):
            out[name] = (math.nan, math.nan, math.nan)
            continue
        spread = max(est - lo, hi - est, 0.0)
        cal = widths.get(name, math.nan)
        half = max(spread, cal) if math.isfinite(cal) else spread
        out[name] = (est, est - half, est + half)
    return out

