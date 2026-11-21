"""step2.py: predict each goal for an unseen flag set, with an uncertainty range."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _load, _encode


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal of train_csv.

    `row` maps input column names to values (goal columns removed). The range
    low..high is the 5th to 95th percentile of out-of-bag prediction errors,
    added around the point estimate.
    """
    rows, inputs, goals = _load(train_csv)
    if not goals:
        return {}

    n = len(rows)
    if n == 0:
        return {name: (float("nan"), float("nan"), float("nan")) for name, _ in goals}

    # Encode training rows and the query together so categorical codes agree.
    query = {c: row.get(c, "") for c in inputs}
    X_all = _encode(rows + [query], inputs)
    X, x_new = X_all[:-1], X_all[-1:]

    out: dict[str, tuple[float, float, float]] = {}
    for name, _sign in goals:
        y = np.array([float(r[name]) for r in rows], dtype=float)

        # Too little data for a meaningful model: fall back to the observed spread.
        if n < 10:
            est = float(y.mean())
            lo = float(y.min())
            hi = float(y.max())
            out[name] = (est, min(lo, est), max(hi, est))
            continue

        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            bootstrap=True,
            oob_score=True,
            random_state=0,
            n_jobs=1,
        )
        model.fit(X, y)
        est = float(model.predict(x_new)[0])

        # Out-of-bag errors estimate how wrong the model is on unseen rows.
        oob = np.asarray(model.oob_prediction_, dtype=float)
        resid = y - oob
        resid = resid[np.isfinite(resid)]
        if resid.size >= 2:
            q_lo = float(np.quantile(resid, 0.05))
            q_hi = float(np.quantile(resid, 0.95))
        else:
            q_lo = q_hi = 0.0

        lo = min(est + q_lo, est)
        hi = max(est + q_hi, est)
        out[name] = (est, lo, hi)

    return out

