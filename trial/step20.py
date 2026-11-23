"""Step 20: race the shortlist and the penalty guesser against standard rivals."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from step1 import _read, _columns, _to_float_column, best
from step2 import _encode_inputs, guess

_TRIALS = 20


def _oriented(rows, goals):
    """Oriented goal matrix (larger is better), NaN where a value is unusable."""
    if not goals or not rows:
        return np.zeros((len(rows), 0))
    return np.column_stack(
        [sign * _to_float_column(rows, name) for name, sign in goals]
    )


def _quality(S):
    """Mean min-max normalised goal value per row; unusable cells count as 0."""
    G = S.shape[1]
    if G == 0:
        return np.zeros(S.shape[0])
    lo = np.zeros(G)
    hi = np.ones(G)
    for g in range(G):
        col = S[:, g]
        fin = col[np.isfinite(col)]
        if fin.size:
            lo[g] = float(fin.min())
            hi[g] = float(fin.max())
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    Z = (S - lo) / span
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z.mean(axis=1)


def _penalty_index(goals):
    """Index of the penalties goal if present, else the first goal."""
    for j, (name, _sign) in enumerate(goals):
        if name.lower().startswith("penalt"):
            return j
    return 0 if goals else None


def _r2(y, p):
    """R^2 over rows where both values are finite; 0.0 if undefined."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() >= 2 and np.var(y[mask]) > 0:
        return float(r2_score(y[mask], p[mask]))
    return 0.0


def race(train_csv, test_csv) -> dict[str, float]:
    """Race our methods against standard rivals on test_csv. Higher is better.

    Shortlist keys (score = mean min-max normalised goal value of the chosen
    rows, in [0, 1]; the shortlist size k is the size of step1.best on test_csv):
      - "ours": step1.best applied to test_csv (Pareto shortlist, goal-based)
      - "top_overall": the k rows with the highest OVR rating
      - "random_subset": mean over repeated random draws of k rows

    Penalty guesser keys (score = R^2 of predicted penalty values on test_csv,
    where the penalty goal is the goal whose name starts with "Penalt"):
      - "guess:ours": step2.guess median estimate
      - "guess:random_forest": scikit-learn RandomForestRegressor

    Note: the shortlist is selected from test rows using their goal values,
    so it is a descriptive comparison of selections, not a prediction from
    training data.
    """
    fieldnames, test_rows = _read(test_csv)
    inputs, goals = _columns(fieldnames)
    n = len(test_rows)
    keys = [
        "ours",
        "top_overall",
        "random_subset",
        "guess:ours",
        "guess:random_forest",
    ]
    if n == 0 or not goals:
        return {k: 0.0 for k in keys}

    scores: dict[str, float] = {}

    # Shortlist versus simple "top overall rating" list.
    S = _oriented(test_rows, goals)
    qual = _quality(S)
    front = list(best(test_csv))
    k = len(front)

    if k == 0:
        scores["ours"] = 0.0
        scores["top_overall"] = 0.0
        scores["random_subset"] = 0.0
    else:
        scores["ours"] = float(qual[np.asarray(front, dtype=int)].mean())

        if "OVR" in fieldnames:
            ovr = _to_float_column(test_rows, "OVR")
            key = np.where(np.isfinite(ovr), ovr, -np.inf)
            order = np.argsort(-key, kind="stable")[:k]
            scores["top_overall"] = float(qual[order].mean())
        else:
            scores["top_overall"] = 0.0

        rs = []
        for t in range(_TRIALS):
            rng = np.random.default_rng(t)
            pick = rng.choice(n, size=k, replace=False)
            rs.append(float(qual[pick].mean()))
        scores["random_subset"] = float(np.mean(rs))

    # Penalty guesser versus off-the-shelf model.
    pj = _penalty_index(goals)
    pname = goals[pj][0]
    y_test = _to_float_column(test_rows, pname)

    guess_pred = np.full(n, np.nan)
    for i, row in enumerate(test_rows):
        est = guess(train_csv, row)
        guess_pred[i] = est[pname][0]
    scores["guess:ours"] = _r2(y_test, guess_pred)

    train_fieldnames, train_rows = _read(train_csv)
    train_inputs, _train_goals = _columns(train_fieldnames)
    rf_pred = np.full(n, np.nan)
    if train_rows:
        if train_inputs:
            X, encode_row = _encode_inputs(train_rows, train_inputs)
            Xte = np.vstack([encode_row(r) for r in test_rows]).astype(float)
        else:
            X = np.zeros((len(train_rows), 1))
            Xte = np.zeros((n, 1))
        yt = _to_float_column(train_rows, pname)
        fin = np.isfinite(yt)
        if fin.sum() >= 2:
            model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
            model.fit(X[fin], yt[fin])
            rf_pred = model.predict(Xte)
    scores["guess:random_forest"] = _r2(y_test, rf_pred)

    return {k: float(v) for k, v in scores.items()}

