"""Step 30: client-facing race for targeting and response guessing.

race(train_csv, test_csv): scores each method on test_csv. Lower is better for
every score. Each score is a normalised rank in [0, 1] (0 = best, 1 = worst).

Client questions answered by the keys:
  * "Does our targeting beat simply mailing our highest-income customers?"
      compare "ours" with "top_income".
  * "Does our response guesser beat an off-the-shelf model?"
      compare "response_ours" with "response_sklearn".
"""

import csv
import math
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import step22
import step27

BUDGET = 50
INCOME = "INCOME"
RESPONSE = "RESPONSE+"


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
    """Every goal as a minimisation value; '+' goals negated, missing = +inf."""
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
    """Score targeting and response methods on test_csv (lower is better).

    Targeting keys (composite normalised rank over all goals, budget 50):
      "ours"           step27 targeting, chosen through a label budget of 50.
      "top_income"     best measured build among the 50 highest-INCOME test rows.
      "random_search"  50 random test rows, best measured one wins.
      "sklearn_forest" off-the-shelf per-goal forest, best predicted composite.

    Response keys (normalised rank on RESPONSE+ alone):
      "response_ours"    step22 guesser: highest predicted RESPONSE+.
      "response_sklearn" off-the-shelf forest on RESPONSE+ alone.
      "response_random"  50 random test rows, best actual RESPONSE+ wins.

    Missing values count as worst. Any score that cannot be computed is NaN.
    """
    nan = math.nan
    keys = ("ours", "top_income", "random_search", "sklearn_forest",
            "response_ours", "response_sklearn", "response_random")
    cols_te, test_rows = _load(test_csv)
    cols_tr, train_rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols_te)
    goals = goal_min + goal_max
    signs = [1.0 if g.endswith("-") else -1.0 for g in goals]
    m = len(test_rows)

    if m == 0 or not goals:
        return {k: nan for k in keys}

    scores: dict[str, float] = {}

    G_true = _min_matrix(test_rows, goal_min, goal_max)
    comp_true = _composite(G_true)

    # ---- ours: step27 targeting through the label budget ----
    def label(i):
        return {g: test_rows[i].get(g) for g in goals}

    with tempfile.TemporaryDirectory() as tmp:
        blind_path = os.path.join(tmp, "blind.csv")
        with open(blind_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols_te)
            writer.writeheader()
            for r in test_rows:
                br = dict(r)
                for g in goals:
                    br[g] = "?"
                writer.writerow(br)
        pick = step27.choose(blind_path, label, budget=BUDGET)
    scores["ours"] = float(comp_true[pick])

    # ---- top_income: highest-income test rows, best measured wins ----
    if INCOME in cols_te:
        inc = np.array([_to_float(r.get(INCOME)) for r in test_rows], dtype=float)
        inc = np.where(np.isfinite(inc), inc, -np.inf)
        top = np.argsort(-inc, kind="stable")[:min(BUDGET, m)]
        scores["top_income"] = float(comp_true[top[int(np.argmin(comp_true[top]))]])
    else:
        scores["top_income"] = nan

    # ---- random_search: same budget, random test rows ----
    rng = np.random.default_rng(0)
    sample = rng.permutation(m)[:min(BUDGET, m)]
    pick = int(sample[int(np.argmin(comp_true[sample]))])
    scores["random_search"] = float(comp_true[pick])

    # ---- sklearn_forest: off-the-shelf per-goal model, composite ----
    X_tr = _features(train_rows, inputs)
    X_te = _features(test_rows, inputs)
    P_rf = np.empty((m, len(goals)), dtype=float)
    for j, g in enumerate(goals):
        y = np.array([_to_float(r.get(g)) for r in train_rows], dtype=float)
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
        P_rf[:, j] = signs[j] * pred
    pick = int(np.argmin(_composite(P_rf)))
    scores["sklearn_forest"] = float(comp_true[pick])

    # ---- response goal: only if RESPONSE+ is a goal in the test file ----
    if RESPONSE in goals:
        resp_min = np.array(
            [(-_to_float(r.get(RESPONSE))) if math.isfinite(_to_float(r.get(RESPONSE)))
             else math.inf for r in test_rows],
            dtype=float,
        )
        resp_rank = _composite(resp_min[:, None])

        # response_ours: step22 guesser, highest predicted RESPONSE+.
        est = np.empty(m, dtype=float)
        for i, r in enumerate(test_rows):
            e = step22.guess(train_csv, r).get(RESPONSE, (nan, nan, nan))[0]
            est[i] = _to_float(e)
        pred_min = np.where(np.isfinite(est), -est, np.inf)
        scores["response_ours"] = float(resp_rank[int(np.argmin(pred_min))])

        # response_sklearn: off-the-shelf forest on RESPONSE+ alone.
        y = np.array([_to_float(r.get(RESPONSE)) for r in train_rows], dtype=float)
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
        scores["response_sklearn"] = float(resp_rank[int(np.argmin(-pred))])

        # response_random: same budget, random test rows, best actual wins.
        rs = rng.permutation(m)[:min(BUDGET, m)]
        pick = int(rs[int(np.argmin(resp_rank[rs]))])
        scores["response_random"] = float(resp_rank[pick])
    else:
        scores["response_ours"] = nan
        scores["response_sklearn"] = nan
        scores["response_random"] = nan

    return {k: float(scores.get(k, nan)) for k in keys}

