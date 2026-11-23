import csv
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict


_N_TREES = 300
_MIN_LEAF = 2
_SEED = 0
_LOW_Q = 0.05
_HIGH_Q = 0.95
_ID_FRACTION = 0.9  # a column this unique is treated as an identifier


def _kind(name):
    if not name:
        return "ignore"
    last = name[-1]
    if last == "X":
        return "ignore"
    if last == "+":
        return "max"
    if last == "-":
        return "min"
    return "input"


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


def _num(cell):
    """Return cell as a finite float, or None if missing or not numeric."""
    if cell is None:
        return None
    try:
        f = float(cell)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def _key(v):
    return "" if _is_missing(v) else str(v)


def _usable_inputs(header, rows):
    """Input columns, minus identifier-like columns (names, urls, row numbers)."""
    n = len(rows)
    keep = []
    for c in header:
        if _kind(c) != "input":
            continue
        raw = [r.get(c) for r in rows]
        present = [v for v in raw if not _is_missing(v)]
        nums = [_num(v) for v in present]
        numeric = bool(present) and all(v is not None for v in nums)
        distinct = len(set(_key(v) for v in raw))
        if n > 0 and distinct > _ID_FRACTION * n:
            if not numeric or all(float(v).is_integer() for v in nums):
                continue
        keep.append(c)
    return keep


def _spec(column, values):
    """(column, labels or None, fill). labels is None for numeric columns."""
    present = [v for v in values if not _is_missing(v)]
    nums = [_num(v) for v in present]
    numeric = bool(present) and all(v is not None for v in nums)
    if numeric:
        fill = float(np.mean(nums))
        return (column, None, fill)
    labels = {k: i for i, k in enumerate(sorted(set(_key(v) for v in values)))}
    return (column, labels, 0.0)


def _encode_row(row, specs):
    out = np.empty(len(specs), dtype=float)
    for j, (c, labels, fill) in enumerate(specs):
        v = row.get(c)
        if labels is None:
            f = _num(v)
            out[j] = fill if f is None else f
        else:
            out[j] = labels.get(_key(v), float(len(labels)))
    return out


def _make_model():
    return RandomForestRegressor(
        n_estimators=_N_TREES,
        min_samples_leaf=_MIN_LEAF,
        random_state=_SEED,
        n_jobs=1,
    )


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a build whose goal columns are unknown.

    Returns {goal: (estimate, low, high)}. The estimate comes from a random
    forest fitted on the training builds' input columns. The low..high range
    is the estimate shifted by the 5th and 95th percentiles of out-of-fold
    residuals, so it covers roughly 90% of the errors seen on the training
    data. Goal values in `row`, if any, are ignored.
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

        if len(specs) == 0 or len(keep) < 4:
            est = float(np.median(y))
            low = float(np.quantile(y, _LOW_Q))
            high = float(np.quantile(y, _HIGH_Q))
            result[g] = (est, min(low, est), max(high, est))
            continue

        k = min(5, len(keep))
        cv = KFold(n_splits=k, shuffle=True, random_state=_SEED)
        oof = cross_val_predict(_make_model(), Xg, y, cv=cv)
        residuals = y - oof
        r_lo, r_hi = np.quantile(residuals, [_LOW_Q, _HIGH_Q])

        model = _make_model().fit(Xg, y)
        est = float(model.predict(xq)[0])
        low = est + float(r_lo)
        high = est + float(r_hi)
        result[g] = (est, min(low, est), max(high, est))

    return result

