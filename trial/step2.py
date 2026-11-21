import csv

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from step1 import _load


def _columns(train_csv):
    with open(train_csv, newline="") as f:
        cols = next(csv.reader(f), [])
    goals = [c for c in cols if c.endswith("+") or c.endswith("-")]
    inputs = [c for c in cols
              if not (c.endswith("+") or c.endswith("-") or c.endswith("X"))]
    return inputs, goals


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate each goal for a flag set, with an interval calibrated on held-out errors."""
    inputs, goals = _columns(train_csv)
    if not goals:
        return {}

    d = _load(train_csv)
    rows = d["rows"]
    n = len(rows)

    X = np.array([[float(r[c]) for c in inputs] for r in rows], dtype=float)
    X = X.reshape(n, len(inputs))
    Y = np.array([[float(r[g]) for g in goals] for r in rows], dtype=float)
    Y = Y.reshape(n, len(goals))

    x_new = np.array(
        [[_as_float(row.get(c, 0.0)) for c in inputs]], dtype=float
    ).reshape(1, len(inputs))

    if n == 0:
        return {g: (0.0, 0.0, 0.0) for g in goals}
    if n < 4:
        # Too little data for a model; report the observed spread.
        out = {}
        for j, g in enumerate(goals):
            lo, mid, hi = float(Y[:, j].min()), float(Y[:, j].mean()), float(Y[:, j].max())
            out[g] = (mid, lo, hi)
        return out

    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
    )
    if len(inputs) == 0:
        est = Y.mean(axis=0).reshape(1, -1)
        oof = np.tile(Y.mean(axis=0), (n, 1))
    else:
        model.fit(X, Y)
        est = model.predict(x_new).reshape(1, -1)
        cv = KFold(n_splits=min(5, n), shuffle=True, random_state=0)
        oof = cross_val_predict(
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
            ),
            X, Y, cv=cv,
        ).reshape(n, len(goals))

    resid = Y - oof
    lo_r = np.quantile(resid, 0.10, axis=0)
    hi_r = np.quantile(resid, 0.90, axis=0)

    result = {}
    for j, g in enumerate(goals):
        e = float(est[0, j])
        lo = e + float(lo_r[j])
        hi = e + float(hi_r[j])
        lo, hi = min(lo, e), max(hi, e)
        result[g] = (e, lo, hi)
    return result


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

