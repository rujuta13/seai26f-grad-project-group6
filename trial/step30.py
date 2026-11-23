"""Step 30: race targeting and response guessing against standard rivals."""

import csv
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from step1 import _read, _columns, _to_float_column
from step2 import _encode_inputs
from step22 import guess as _guess22
from step27 import choose as _choose

_TRIALS = 20


def _response_name(goals):
    """The response goal (name starts with RESPONSE), or None."""
    for name, _sign in goals:
        if name.upper().startswith("RESPONSE"):
            return name
    return None


def _norm01(y):
    """Min-max normalise finite values to [0, 1]; unusable cells count as 0."""
    fin = y[np.isfinite(y)]
    if fin.size == 0:
        return np.zeros(len(y))
    lo = float(fin.min())
    hi = float(fin.max())
    span = hi - lo if hi - lo > 0 else 1.0
    z = (y - lo) / span
    return np.where(np.isfinite(z), z, 0.0)


def _r2(y, p):
    """R^2 over rows where both values are finite; 0.0 if undefined."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() >= 2 and np.var(y[mask]) > 0:
        return float(r2_score(y[mask], p[mask]))
    return 0.0


def _targeting_scores(test_csv, fieldnames, test_rows, names, target):
    """Targeting keys: normalised response of the single row each method picks."""
    n = len(test_rows)
    scores = {"ours": 0.0, "random_search": 0.0, "highest_income": 0.0}
    if target is None or n == 0:
        return scores

    y = _to_float_column(test_rows, target)
    tnorm = _norm01(y)
    B = min(30, n)

    # Our targeting: labelled search via step27.choose on a blinded copy.
    def label(i):
        return {g: test_rows[i].get(g) for g in names}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blind.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in test_rows:
                rr = {c: r.get(c) for c in fieldnames}
                for g in names:
                    rr[g] = "?"
                w.writerow(rr)
        ours_idx = _choose(path, label, budget=B, goal=target)
    scores["ours"] = float(tnorm[ours_idx])

    # Random search rival: best of B random rows by the target, averaged over draws.
    rs = []
    for t in range(_TRIALS):
        rng = np.random.default_rng(t)
        pick = rng.choice(n, size=B, replace=False)
        rs.append(float(np.max(tnorm[pick])))
    scores["random_search"] = float(np.mean(rs))

    # Baseline: mail the highest-income customer.
    if "INCOME" in fieldnames:
        inc = _to_float_column(test_rows, "INCOME")
        key = np.where(np.isfinite(inc), inc, -np.inf)
        scores["highest_income"] = float(tnorm[int(np.argmax(key))])

    return scores


def _guesser_scores(train_csv, test_rows, goals, target):
    """Response guesser keys: R^2 of predicted response on test rows."""
    out = {"guess:ours": 0.0, "guess:random_forest": 0.0}
    if target is None or not test_rows:
        return out

    y_test = _to_float_column(test_rows, target)

    pred = np.full(len(test_rows), np.nan)
    for i, row in enumerate(test_rows):
        est = _guess22(train_csv, row)
        pred[i] = est[target][0]
    out["guess:ours"] = _r2(y_test, pred)

    fieldnames, train_rows = _read(train_csv)
    inputs, _ = _columns(fieldnames)
    n_test = len(test_rows)
    rf_pred = np.full(n_test, np.nan)
    if train_rows:
        if inputs:
            X, encode_row = _encode_inputs(train_rows, inputs)
            Xte = np.vstack([encode_row(r) for r in test_rows]).astype(float)
        else:
            X = np.zeros((len(train_rows), 1))
            Xte = np.zeros((n_test, 1))
        yt = _to_float_column(train_rows, target)
        fin = np.isfinite(yt)
        if fin.sum() >= 2:
            model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
            model.fit(X[fin], yt[fin])
            rf_pred = model.predict(Xte)
    out["guess:random_forest"] = _r2(y_test, rf_pred)

    return out


def race(train_csv, test_csv) -> dict[str, float]:
    """Race our targeting and response guesser against standard rivals on test_csv.

    Targeting keys (score = normalised response, in [0, 1], of the single customer
    each method picks with a budget of up to 30 labelled rows; the response goal is
    the goal whose name starts with "RESPONSE"):
      - "ours": step27.choose
      - "random_search": best of 30 random customers, averaged over repeated draws
      - "highest_income": mail the customer with the highest INCOME
    Guesser keys (score = R^2 of predicted response on test_csv):
      - "guess:ours": step22.guess
      - "guess:random_forest": scikit-learn RandomForestRegressor
    """
    fieldnames, test_rows = _read(test_csv)
    inputs, goals = _columns(fieldnames)
    names = [g for g, _ in goals]
    target = _response_name(goals)

    keys = [
        "ours",
        "random_search",
        "highest_income",
        "guess:ours",
        "guess:random_forest",
    ]
    if not test_rows or not goals:
        return {k: 0.0 for k in keys}

    scores: dict[str, float] = {}
    scores.update(_targeting_scores(test_csv, fieldnames, test_rows, names, target))
    scores.update(_guesser_scores(train_csv, test_rows, goals, target))
    return {k: float(scores.get(k, 0.0)) for k in keys}

