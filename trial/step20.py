"""Step 20: client-facing race against standard rivals.

race(train_csv, test_csv): scores each method on test_csv. Lower is better
for every score; each is a normalised rank in [0, 1] (0 = best, 1 = worst).

Client questions answered by the keys:
  * "Does our shortlist beat a simple top-overall-rating list?"
      compare "ours" with "top_overall".
  * "Does our penalty guesser beat an off-the-shelf model?"
      compare "penalty_ours" with "penalty_sklearn".
"""

import csv
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import step2
import step10

PEN = "Penalties+"
OVR = "OVR"
BUDGET = 30


def _load(path):
    with open(path, newline="") as f:
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


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[np.asarray(inverse).reshape(-1)]


def _composite(M):
    """Mean normalised rank per row of a minimisation matrix (0 = best, 1 = worst)."""
    m = M.shape[0]
    if m == 0 or M.shape[1] == 0:
        return np.zeros(m, dtype=float)
    b = np.zeros(m, dtype=float)
    for j in range(M.shape[1]):
        b += (_average_ranks(M[:, j]) - 1.0) / max(m - 1, 1)
    return b / M.shape[1]


def _min_matrix(rows, goal_min, goal_max):
    goals = goal_min + goal_max
    G = np.empty((len(rows), len(goals)), dtype=float)
    for j, c in enumerate(goals):
        v = np.array([_to_float(r.get(c)) for r in rows], dtype=float)
        missing = ~np.isfinite(v)
        if c.endswith("+"):
            v = -v
        v[missing] = np.inf
        G[:, j] = v
    return G


def _features(rows, inputs):
    n = len(rows)
    if not inputs:
        return np.zeros((n, 1), dtype=float)
    X = np.array(
        [[_to_float(r.get(c)) for c in inputs] for r in rows], dtype=float
    ).reshape(n, len(inputs))
    return np.where(np.isfinite(X), X, 0.0)


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our methods and the standard rivals on test_csv (lower is better).

    Keys:
      "ours"            step7 shortlist tuner (budget 30), composite over all goals.
      "top_overall"     best measured build among the 30 highest-OVR test rows
                        (same budget of 30 measurements), composite over all goals.
      "random_search"   30 random test builds, best measured wins, composite.
      "sklearn_forest"  off-the-shelf per-goal forest, best predicted composite.
      "penalty_ours"    step2 guesser: build with the highest predicted
                        Penalties+, scored by its Penalties+ normalised rank.
      "penalty_sklearn" off-the-shelf forest on Penalties+ alone, same scoring.
    Missing values count as worst. Any score that cannot be computed is NaN.
    """
    scores: dict[str, float] = dict(step10.race(train_csv, test_csv))

    cols_te, test_rows = _load(test_csv)
    cols_tr, train_rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols_te)
    goals = goal_min + goal_max
    m = len(test_rows)
    nan = math.nan

    extra_keys = ("top_overall", "penalty_ours", "penalty_sklearn")
    if m == 0 or not goals:
        for k in extra_keys:
            scores[k] = nan
        return scores

    G = _min_matrix(test_rows, goal_min, goal_max)
    comp = _composite(G)

    # ---- top_overall: shortlist of the 30 highest OVR rows, best measured wins ----
    if OVR in cols_te:
        ovr = np.array([_to_float(r.get(OVR)) for r in test_rows], dtype=float)
        ovr = np.where(np.isfinite(ovr), ovr, -np.inf)
        top = np.argsort(-ovr, kind="stable")[:min(BUDGET, m)]
        scores["top_overall"] = float(comp[top[int(np.argmin(comp[top]))]])
    else:
        scores["top_overall"] = nan

    # ---- penalty goal: only if the column is a goal in the test file ----
    if PEN in goals:
        pen_min = np.array(
            [(-_to_float(r.get(PEN))) if math.isfinite(_to_float(r.get(PEN)))
             else math.inf for r in test_rows],
            dtype=float,
        )
        pen_rank = _composite(pen_min[:, None])

        # penalty_ours: step2 guesser, highest predicted Penalties+.
        est = np.empty(m, dtype=float)
        for i, r in enumerate(test_rows):
            e = step2.guess(train_csv, r).get(PEN, (nan, nan, nan))[0]
            est[i] = _to_float(e)
        pred_min = np.where(np.isfinite(est), -est, np.inf)
        scores["penalty_ours"] = float(pen_rank[int(np.argmin(pred_min))])

        # penalty_sklearn: off-the-shelf forest on Penalties+ alone.
        X_tr = _features(train_rows, inputs)
        X_te = _features(test_rows, inputs)
        y = np.array([_to_float(r.get(PEN)) for r in train_rows], dtype=float)
        ok = np.isfinite(y)
        if ok.sum() >= 10 and inputs:
            model = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=2, random_state=0
            )
            model.fit(X_tr[ok], y[ok])
            pred = model.predict(X_te)
        else:
            fill = float(np.median(y[ok])) if ok.any() else 0.0
            pred = np.full(m, fill)
        scores["penalty_sklearn"] = float(pen_rank[int(np.argmin(-pred))])
    else:
        scores["penalty_ours"] = nan
        scores["penalty_sklearn"] = nan

    return {k: float(v) for k, v in scores.items()}

