"""Step 2: guess the goal values of an unseen flag set.

Standard library only. Loading, column splitting, input encoding and the
squared-distance helper are reused from step1.

Method: inverse-distance-weighted k-nearest-neighbour regression on the
input columns. Each interval is the estimate shifted by the 5th and 95th
percentiles of leave-one-out residuals measured on the training data, so
the band reflects the error the model actually makes on this data.
"""

import heapq
import math

from step1 import _load, _columns, _input_matrix, _sqdist


K_NEIGHBOURS = 7
CALIB_SAMPLE = 200      # training rows used for leave-one-out calibration
BAND_LO = 0.05          # lower residual quantile for the interval
BAND_HI = 0.95          # upper residual quantile for the interval

_cache = {}


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _predict(query, points, targets, exclude=None):
    """Inverse-distance-weighted mean of the nearest neighbours' goal values.

    Returns None when there is no neighbour to average over."""
    nearest = heapq.nsmallest(
        K_NEIGHBOURS,
        ((_sqdist(query, p), i) for i, p in enumerate(points) if i != exclude),
    )
    if not nearest:
        return None
    m = len(targets[0])
    est = [0.0] * m
    wsum = 0.0
    for d, i in nearest:
        w = 1.0 / (1.0 + d)
        wsum += w
        for g in range(m):
            est[g] += w * targets[i][g]
    return [v / wsum for v in est]


def _calibrate(points, targets, n_goals):
    """Per goal, the (lo, hi) residual quantiles of leave-one-out predictions."""
    n = len(points)
    step = max(1, n // CALIB_SAMPLE)
    resid = [[] for _ in range(n_goals)]
    for i in range(0, n, step):
        pred = _predict(points[i], points, targets, exclude=i)
        if pred is None:
            continue
        for g in range(n_goals):
            resid[g].append(targets[i][g] - pred[g])
    bands = []
    for vals in resid:
        if not vals:
            bands.append((-math.inf, math.inf))
            continue
        vals.sort()
        bands.append((_quantile(vals, BAND_LO), _quantile(vals, BAND_HI)))
    return bands


def _model(train_csv):
    """Load and cache the training data, its encoding and its calibration."""
    if train_csv in _cache:
        return _cache[train_csv]
    header, rows = _load(train_csv)
    inputs, goals = _columns(header)
    points = _input_matrix(rows, inputs)
    targets = [[float(r[name]) for name, _ in goals] for r in rows]
    bands = _calibrate(points, targets, len(goals))
    model = {
        "rows": rows,
        "inputs": inputs,
        "goals": goals,
        "points": points,
        "targets": targets,
        "bands": bands,
    }
    _cache[train_csv] = model
    return model


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for an unseen flag set.

    `row` holds the input columns only (goal columns removed). Returns
    {goal: (estimate, low, high)} where low..high is the calibrated range
    the true value should fall in.
    """
    model = _model(train_csv)
    goals = model["goals"]
    targets = model["targets"]

    if not targets:
        return {name: (0.0, -math.inf, math.inf) for name, _ in goals}

    # Encode the query together with the training rows so that categorical
    # codes match the ones used for the training points.
    probe = {c: row.get(c, 0.0) for c in model["inputs"]}
    qpoint = _input_matrix(model["rows"] + [probe], model["inputs"])[-1]

    pred = _predict(qpoint, model["points"], targets)

    out = {}
    for g, (name, _) in enumerate(goals):
        est = pred[g]
        rlo, rhi = model["bands"][g]
        out[name] = (est, min(est, est + rlo), max(est, est + rhi))
    return out

