import csv

import numpy as np
from sklearn.cluster import KMeans


def _read(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _columns(fieldnames):
    # goals: (name, sign) with sign +1 to maximise, -1 to minimise
    goals = [(c, 1.0) for c in fieldnames if c.endswith("+")]
    goals += [(c, -1.0) for c in fieldnames if c.endswith("-")]
    inputs = [c for c in fieldnames if not c.endswith(("+", "-", "X"))]
    return inputs, goals


def _to_float_column(rows, col):
    """Convert a column to floats; non-numeric strings are coded by first appearance."""
    values = [r[col] for r in rows]
    try:
        return np.array([float(v) for v in values], dtype=float)
    except (TypeError, ValueError):
        codes = {}
        out = np.empty(len(values), dtype=float)
        for i, v in enumerate(values):
            if v not in codes:
                codes[v] = float(len(codes))
            out[i] = codes[v]
        return out


def families(train_csv) -> list[int]:
    fieldnames, rows = _read(train_csv)
    n = len(rows)
    if n == 0:
        return []
    inputs, _ = _columns(fieldnames)
    if not inputs:
        return [0] * n
    X = np.column_stack([_to_float_column(rows, c) for c in inputs])
    k = min(5, n)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    return [int(x) for x in labels]


def best(train_csv) -> list[int]:
    fieldnames, rows = _read(train_csv)
    n = len(rows)
    _, goals = _columns(fieldnames)
    cap = int(np.floor(0.05 * n))
    if n == 0 or cap == 0 or not goals:
        return []

    # Oriented goal matrix: larger is better on every column.
    S = np.column_stack(
        [sign * _to_float_column(rows, name) for name, sign in goals]
    )

    # Pareto front: a row is kept unless some other row is at least as good
    # on every goal and strictly better on at least one.
    nondominated = np.ones(n, dtype=bool)
    for i in range(n):
        at_least = np.all(S >= S[i], axis=1)
        strictly = np.any(S > S[i], axis=1)
        if np.any(at_least & strictly):
            nondominated[i] = False
    front = np.flatnonzero(nondominated)

    if len(front) > cap:
        # Rank front rows by min-max normalised goal sum and keep the top `cap`.
        F = S[front]
        lo = S.min(axis=0)
        hi = S.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        score = ((F - lo) / span).sum(axis=1)
        order = np.argsort(-score, kind="stable")
        front = front[order[:cap]]

    return sorted(int(i) for i in front)

