import csv
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a flag set `row` (goal columns removed).

    Returns {goal: (estimate, low, high)} where low..high is an approximate
    90% interval built from out-of-bag residuals of a random forest.
    """
    cols, rows = _load(train_csv)
    goals = [c for c in cols if c.endswith(("+", "-"))]
    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]

    result: dict[str, tuple[float, float, float]] = {}

    if not rows or not goals:
        for g in goals:
            result[g] = (0.0, -math.inf, math.inf)
        return result

    # Feature matrix for training rows; missing cells become 0.0 (as in step1).
    X = np.array(
        [[_to_float(r.get(c, math.nan)) for c in inputs] for r in rows],
        dtype=float,
    ).reshape(len(rows), len(inputs))
    X = np.where(np.isnan(X), 0.0, X)

    # Feature vector for the query row.
    x = np.array(
        [[_to_float(row.get(c, math.nan)) for c in inputs]], dtype=float
    ).reshape(1, len(inputs))
    x = np.where(np.isnan(x), 0.0, x)

    for g in goals:
        y = np.array([_to_float(r.get(g, math.nan)) for r in rows], dtype=float)
        finite = np.isfinite(y)
        Xg, yg = X[finite], y[finite]

        if len(yg) == 0:
            result[g] = (0.0, -math.inf, math.inf)
            continue

        if len(yg) < 10 or not inputs:
            # Too little data for a model: use the empirical distribution.
            est = float(np.median(yg))
            lo = float(np.quantile(yg, 0.05))
            hi = float(np.quantile(yg, 0.95))
            result[g] = (est, min(lo, est), max(hi, est))
            continue

        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        )
        model.fit(Xg, yg)

        est = float(model.predict(x)[0])

        oob = np.asarray(model.oob_prediction_, dtype=float)
        ok = np.isfinite(oob)
        if ok.sum() >= 10:
            resid = yg[ok] - oob[ok]
            q_lo, q_hi = np.quantile(resid, [0.05, 0.95])
        else:
            q_lo, q_hi = np.quantile(yg, [0.05, 0.95]) - np.median(yg)

        low = est + float(q_lo)
        high = est + float(q_hi)
        low = min(low, est)
        high = max(high, est)
        result[g] = (est, low, high)

    return result

