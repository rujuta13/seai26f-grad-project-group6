import csv
import math
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import step2
import step7


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
    """Ranks 1..N with ties replaced by their average rank."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[inverse]


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
    """Compare the client's tuner and guesser with standard rivals on test_csv.

    Every method picks one row of test_csv. Its score is the chosen build's
    composite normalised rank over all test rows (mean rank across goals,
    0 = best, 1 = worst), so LOWER IS BETTER. Missing goal values count as worst.

    Methods:
      "ours"             step7 tuner: label budget of 30 test builds.
      "ours_guesser"     step2 guesser: predicts every goal per test build,
                         picks the build with the best predicted composite.
      "random_search"    30 test builds drawn at random, best measured one wins.
      "sklearn_forest"   off-the-shelf RandomForestRegressor per goal, trained
                         on train_csv, picks the best predicted composite.
    """
    cols_tr, train_rows = _load(train_csv)
    cols_te, test_rows = _load(test_csv)
    inputs, goal_max, goal_min = _split_columns(cols_te)
    goals = goal_min + goal_max
    signs = [1.0 if g.endswith("-") else -1.0 for g in goals]
    m = len(test_rows)
    budget = 30

    if m == 0 or not goals:
        return {k: math.nan for k in ("ours", "ours_guesser",
                                      "random_search", "sklearn_forest")}

    # Ground truth on the test set, used only to score choices.
    G_true = _min_matrix(test_rows, goal_min, goal_max)
    comp_true = _composite(G_true)

    scores: dict[str, float] = {}

    # ---- ours: step7 tuner, measured through the label budget ----
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
        pick = step7.choose(blind_path, label, budget=budget)
    scores["ours"] = float(comp_true[pick])

    # ---- ours_guesser: step2 predictions for every test build ----
    P_guess = np.empty((m, len(goals)), dtype=float)
    for i, r in enumerate(test_rows):
        est = step2.guess(train_csv, r)
        for j, g in enumerate(goals):
            e = est.get(g, (math.nan, 0.0, 0.0))[0]
            P_guess[i, j] = signs[j] * e
    pick = int(np.argmin(_composite(P_guess)))
    scores["ours_guesser"] = float(comp_true[pick])

    # ---- random_search: same budget, random test builds ----
    rng = np.random.default_rng(0)
    sample = rng.permutation(m)[:min(budget, m)]
    pick = int(sample[int(np.argmin(comp_true[sample]))])
    scores["random_search"] = float(comp_true[pick])

    # ---- sklearn_forest: off-the-shelf model trained on train_csv ----
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

    return scores

