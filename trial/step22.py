import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from step12 import (
    _encode_row,
    _kind,
    _load,
    _num,
    _spec,
    _usable_inputs,
)


_N_TREES = 300
_MIN_LEAF = 2
_SEED = 0
_LOW_Q = 0.05
_HIGH_Q = 0.95


def _regressor():
    return RandomForestRegressor(
        n_estimators=_N_TREES,
        min_samples_leaf=_MIN_LEAF,
        random_state=_SEED,
        n_jobs=1,
    )


def _classifier():
    return RandomForestClassifier(
        n_estimators=_N_TREES,
        min_samples_leaf=_MIN_LEAF,
        random_state=_SEED,
        n_jobs=1,
    )


def _regression_range(Xg, y, xq):
    """Estimate and range for a goal taking many values (e.g. purchase counts)."""
    ymin, ymax = float(y.min()), float(y.max())
    if Xg.shape[1] == 0 or len(y) < 4:
        est = float(np.median(y))
        low = float(np.quantile(y, _LOW_Q))
        high = float(np.quantile(y, _HIGH_Q))
    else:
        k = min(5, len(y))
        cv = KFold(n_splits=k, shuffle=True, random_state=_SEED)
        oof = cross_val_predict(_regressor(), Xg, y, cv=cv)
        r_lo, r_hi = np.quantile(y - oof, [_LOW_Q, _HIGH_Q])
        est = float(_regressor().fit(Xg, y).predict(xq)[0])
        low = est + float(r_lo)
        high = est + float(r_hi)
    # Goal counts cannot leave the range seen in training.
    est = min(max(est, ymin), ymax)
    low = min(max(low, ymin), ymax)
    high = min(max(high, ymin), ymax)
    low, high = min(low, est), max(high, est)
    return (est, low, high)


def _binary_range(Xg, y, xq, lo, hi, counts):
    """Estimate and range for a goal with exactly two values (e.g. RESPONSE).

    The estimate is the expected value lo + P(y == hi) * (hi - lo), with the
    probability from a random forest classifier. The range is the estimate
    shifted by out-of-fold residual quantiles and kept inside [lo, hi].
    """
    t = (y == hi).astype(int)
    k = min(5, int(counts.min()))
    if k < 2:
        return _regression_range(Xg, y, xq)

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=_SEED)
    oof_p = cross_val_predict(
        _classifier(), Xg, t, cv=cv, method="predict_proba"
    )[:, 1]
    oof_val = lo + oof_p * (hi - lo)
    r_lo, r_hi = np.quantile(y - oof_val, [_LOW_Q, _HIGH_Q])

    model = _classifier().fit(Xg, t)
    p = float(model.predict_proba(xq)[0][1])
    est = lo + p * (hi - lo)
    low = min(max(est + float(r_lo), lo), hi)
    high = min(max(est + float(r_hi), lo), hi)
    est = min(max(est, lo), hi)
    low, high = min(low, est), max(high, est)
    return (float(est), float(low), float(high))


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a new customer whose goal columns are unknown.

    Returns {goal: (estimate, low, high)}. Goals with exactly two observed
    values (such as a yes/no campaign response) are estimated with a classifier
    as an expected value between those two values. Other goals (such as
    purchase counts) are estimated with a random forest regressor. The low..high
    range covers roughly 90% of out-of-fold errors on the training builds, and
    is kept within the range of values seen in training. Goal values in `row`,
    if any, are ignored.
    """
    header, train = _load(train_csv)
    goals = [c for c in header if _kind(c) in ("max", "min")]
    if not goals:
        return {}

    n = len(train)
    inputs = _usable_inputs(header, train)
    specs = [_spec(c, [r.get(c) for r in train]) for c in inputs]

    if n > 0 and specs:
        X = np.vstack([_encode_row(r, specs) for r in train])
    else:
        X = np.zeros((n, len(specs)), dtype=float)
    xq = _encode_row(row, specs).reshape(1, len(specs))

    result = {}
    for g in goals:
        ys = [_num(r.get(g)) for r in train]
        keep = [i for i, v in enumerate(ys) if v is not None]
        if not keep:
            result[g] = (0.0, 0.0, 0.0)
            continue

        y = np.asarray([ys[i] for i in keep], dtype=float)
        Xg = X[keep]
        values = np.unique(y)

        if len(values) == 1:
            v = float(values[0])
            result[g] = (v, v, v)
        elif len(values) == 2:
            lo, hi = float(values[0]), float(values[1])
            counts = np.bincount((y == hi).astype(int), minlength=2)
            result[g] = _binary_range(Xg, y, xq, lo, hi, counts)
        else:
            result[g] = _regression_range(Xg, y, xq)

    return result

