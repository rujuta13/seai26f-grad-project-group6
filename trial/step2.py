import csv

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict


def _to_value(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def _column_kind(name):
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
        rows = [{k: _to_value(v) for k, v in r.items()} for r in reader]
    return header, rows


def _build_encoders(values_by_col):
    """Per input column: None for numeric, else a label -> code dict."""
    encoders = {}
    for c, values in values_by_col.items():
        if all(isinstance(v, float) for v in values):
            encoders[c] = None
        else:
            labels = {}
            for v in values:
                key = str(v)
                if key not in labels:
                    labels[key] = float(len(labels))
            encoders[c] = labels
    return encoders


def _encode(value, labels):
    if labels is None:
        if isinstance(value, float):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1.0
    if value is None:
        return -1.0
    return labels.get(str(value), -1.0)


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    header, rows = _load(train_csv)
    goals = [c for c in header if _column_kind(c) in ("max", "min")]
    inputs = [c for c in header if _column_kind(c) == "input"]
    n = len(rows)

    if n == 0 or not goals:
        return {g: (0.0, 0.0, 0.0) for g in goals}

    values_by_col = {c: [r.get(c) for r in rows] for c in inputs}
    encoders = _build_encoders(values_by_col)

    X = np.zeros((n, len(inputs)), dtype=float)
    for j, c in enumerate(inputs):
        labels = encoders[c]
        X[:, j] = [_encode(v, labels) for v in values_by_col[c]]

    xq = np.array(
        [[_encode(row.get(c), encoders[c]) for c in inputs]], dtype=float
    ).reshape(1, len(inputs))

    result = {}
    for g in goals:
        y = np.asarray([float(r[g]) for r in rows], dtype=float)

        if len(inputs) == 0 or n < 4:
            est = float(np.median(y))
            low = float(np.quantile(y, 0.05))
            high = float(np.quantile(y, 0.95))
            result[g] = (est, min(low, est), max(high, est))
            continue

        def make_model():
            return RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=0,
                n_jobs=1,
            )

        k = min(5, n)
        cv = KFold(n_splits=k, shuffle=True, random_state=0)
        oof = cross_val_predict(make_model(), X, y, cv=cv)
        residuals = y - oof
        r_lo, r_hi = np.quantile(residuals, [0.05, 0.95])

        model = make_model().fit(X, y)
        est = float(model.predict(xq)[0])
        low = est + float(r_lo)
        high = est + float(r_hi)
        result[g] = (est, min(low, est), max(high, est))

    return result

