import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _column_kind, _goal_matrix, _input_matrix, _load


def _badness(G):
    """Per-row badness in [0, 1]: mean normalised rank across goals (1 = worst)."""
    n = G.shape[0]
    ranks = G.argsort(axis=0).argsort(axis=0).astype(float)
    return ranks.mean(axis=1) / max(n - 1, 1)


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Each row gets a badness score (mean normalised rank over all goals, where
    larger is worse). A random forest is fitted from the input columns to that
    score, and inputs are ordered by their impurity-based importance. Ties are
    broken by header order.
    """
    header, rows = _load(train_csv)
    inputs = [c for c in header if _column_kind(c) == "input"]
    n = len(rows)

    if n == 0 or not inputs:
        return list(inputs)

    G = _goal_matrix(header, rows)
    if G.shape[1] == 0:
        return list(inputs)

    X = _input_matrix(header, rows)
    if X.shape[1] == 0:
        return []

    score = _badness(G)
    if n < 2 or float(score.std()) == 0.0:
        return list(inputs)

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=1,
    ).fit(X, score)
    importance = np.asarray(model.feature_importances_, dtype=float)

    order = sorted(range(len(inputs)), key=lambda j: (-importance[j], j))
    return [inputs[j] for j in order]

