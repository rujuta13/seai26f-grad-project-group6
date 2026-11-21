import csv
import itertools
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _split_columns(cols):
    goal_max = [c for c in cols if c.endswith("+")]
    goal_min = [c for c in cols if c.endswith("-")]
    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]
    return inputs, goal_max, goal_min


def fix(train_csv, row, frozen=()) -> dict:
    """Return the inputs of `row` with as few values changed as possible so that
    the build is rated good (see step6.why: on average at most 25% of training
    builds do better on each goal). Columns named in `frozen` never change.

    Goal values for candidate rows are predicted with one random forest per goal
    (as in step2). The search tries 0, 1, 2, ... changed inputs in turn and
    stops at the first size that yields a good candidate. If no candidate within
    MAX_CHANGES is good, the best-rated candidate found is returned.
    """
    cols, rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max
    frozen = set(frozen)

    out = {c: row.get(c) for c in inputs}
    n = len(rows)
    if n == 0 or not goals or not inputs:
        return out

    # Training feature matrix and query vector (missing cells -> 0.0, as in step1/2).
    X = np.array(
        [[_to_float(r.get(c)) for c in inputs] for r in rows], dtype=float
    ).reshape(n, len(inputs))
    X = np.where(np.isnan(X), 0.0, X)

    x0 = np.array([_to_float(row.get(c)) for c in inputs], dtype=float)
    x0 = np.where(np.isnan(x0), 0.0, x0)

    # Minimisation goal matrix: '+' goals negated, missing = +inf (worst).
    G = np.empty((n, len(goals)), dtype=float)
    signs = []
    for j, g in enumerate(goals):
        s = 1.0 if g.endswith("-") else -1.0
        signs.append(s)
        v = np.array([_to_float(r.get(g)) for r in rows], dtype=float)
        miss = ~np.isfinite(v)
        v = s * v
        v[miss] = np.inf
        G[:, j] = v
    sorted_cols = [np.sort(G[:, j]) for j in range(len(goals))]

    # One predictor per goal.
    models = []
    for g in goals:
        raw = np.array([_to_float(r.get(g)) for r in rows], dtype=float)
        ok = np.isfinite(raw)
        if ok.sum() >= 10:
            m = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=2, random_state=0
            )
            m.fit(X[ok], raw[ok])
            models.append(m)
        else:
            models.append(float(np.median(raw[ok])) if ok.any() else 0.0)

    def predict_min(Xc):
        P = np.empty((Xc.shape[0], len(goals)), dtype=float)
        for j, m in enumerate(models):
            if isinstance(m, float):
                pred = np.full(Xc.shape[0], m)
            else:
                pred = m.predict(Xc)
            P[:, j] = signs[j] * pred
        return P

    def rating(P):
        """Average fraction of training builds that beat each candidate (ties half)."""
        fr = np.zeros(P.shape[0], dtype=float)
        for j in range(len(goals)):
            sc = sorted_cols[j]
            lt = np.searchsorted(sc, P[:, j], side="left")
            le = np.searchsorted(sc, P[:, j], side="right")
            fr += (lt + 0.5 * (le - lt)) / n
        return fr / len(goals)

    GOOD = 0.25
    MAX_CHANGES = 4
    MAX_CANDIDATES = 20000

    # Columns that may change, with their alternative values from training data.
    eligible = []
    for idx, c in enumerate(inputs):
        if c in frozen:
            continue
        alts = sorted({float(v) for v in X[:, idx]} - {float(x0[idx])})
        if alts:
            eligible.append((idx, alts))

    p0 = float(rating(predict_min(x0[None, :]))[0])
    if p0 <= GOOD:
        return out

    best_p = p0
    best_x = x0.copy()

    found = None
    for k in range(1, MAX_CHANGES + 1):
        cands = []
        for combo in itertools.combinations(eligible, k):
            value_lists = [alts for _, alts in combo]
            for vals in itertools.product(*value_lists):
                x = x0.copy()
                for (idx, _), v in zip(combo, vals):
                    x[idx] = v
                cands.append(x)
                if len(cands) >= MAX_CANDIDATES:
                    break
            if len(cands) >= MAX_CANDIDATES:
                break
        if not cands:
            break

        C = np.array(cands, dtype=float)
        p = rating(predict_min(C))
        i = int(np.argmin(p))  # ties resolve to the first candidate found
        if p[i] < best_p:
            best_p = float(p[i])
            best_x = C[i].copy()
        if p[i] <= GOOD:
            found = C[i].copy()
            break

    chosen = found if found is not None else best_x

    result = {}
    for idx, c in enumerate(inputs):
        if chosen[idx] == x0[idx]:
            result[c] = row.get(c)
        else:
            v = float(chosen[idx])
            result[c] = int(v) if v.is_integer() else v
    return result

