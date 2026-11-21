import csv
import math

import numpy as np
from sklearn.cluster import KMeans


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


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    cols, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []

    inputs, _, _ = _split_columns(cols)
    if not inputs:
        return [0] * n

    X = np.array(
        [[_to_float(r[c]) for c in inputs] for r in rows], dtype=float
    )
    X = np.where(np.isnan(X), 0.0, X)

    n_unique = len(np.unique(X, axis=0))
    k = min(5, n_unique)
    if k <= 1:
        return [0] * n

    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = model.fit_predict(X)
    return [int(x) for x in labels]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows."""
    cols, rows = _load(train_csv)
    n = len(rows)
    cap = int(math.floor(0.05 * n))
    if n == 0 or cap == 0:
        return []

    _, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max
    if not goals:
        return []

    # Build a matrix where every goal is minimised.
    G = np.empty((n, len(goals)), dtype=float)
    for j, c in enumerate(goals):
        vals = np.array([_to_float(r[c]) for r in rows], dtype=float)
        vals = np.where(np.isnan(vals), np.inf, vals)  # missing = worst
        if c.endswith("+"):
            vals = -vals
        G[:, j] = vals

    # Pareto front (non-dominated rows), all goals minimised.
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        weakly = np.all(G <= G[i], axis=1)
        strictly = np.any(G < G[i], axis=1)
        dominated[i] = np.any(weakly & strictly)
    front = np.flatnonzero(~dominated)

    if len(front) > cap:
        # Too many non-dominated rows: keep those with the best average rank
        # across goals.
        sub = G[front]
        rank_sum = np.zeros(len(front), dtype=float)
        for j in range(sub.shape[1]):
            order = np.argsort(sub[:, j], kind="stable")
            ranks = np.empty(len(front), dtype=float)
            ranks[order] = np.arange(len(front), dtype=float)
            rank_sum += ranks
        keep = np.argsort(rank_sum, kind="stable")[:cap]
        front = front[keep]

    return sorted(int(i) for i in front)

