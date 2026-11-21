"""step2.py: guess the goal values of an unseen flag set from the training data.

Reuses the loading and column-classification helpers from step1. The guess is a
k-nearest-neighbour estimate in standardised input space: the estimate is the
median goal value of the nearest training rows, and the range is their 10th to
90th percentile.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _load, _to_value

_K = 15          # number of neighbours used for the guess
_LOW_Q = 0.10    # lower bound of the stated range
_HIGH_Q = 0.90   # upper bound of the stated range


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of `row` as (estimate, low, high).

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns {goal column name: (estimate, low, high)} for every goal in the
    training file, in header order.
    """
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")

    _, _, inputs = _classify(header)
    goal_cols = [i for i, name in enumerate(header) if name.endswith(("+", "-"))]
    n = len(rows)

    # Encode training inputs, keeping the mappings needed to encode the query row.
    X = np.zeros((n, len(inputs)), dtype=float)
    specs = []
    for j, col in enumerate(inputs):
        vals = [_to_value(r[col]) for r in rows]
        if all(isinstance(v, float) for v in vals):
            X[:, j] = vals
            specs.append(("num", float(np.median(vals))))
        else:
            codes = {v: k for k, v in enumerate(sorted({str(v) for v in vals}))}
            X[:, j] = [codes[str(v)] for v in vals]
            specs.append(("cat", codes))

    # Encode the query row with the same mappings.
    q = np.zeros(len(inputs), dtype=float)
    for j, col in enumerate(inputs):
        raw = row.get(header[col])
        kind, info = specs[j]
        if kind == "num":
            v = _to_value(raw) if raw is not None else None
            q[j] = v if isinstance(v, float) else info
        else:
            q[j] = info.get(str(raw), -1) if raw is not None else -1

    # Standardise, then find the nearest training rows to the query.
    if len(inputs) > 0:
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        zq = scaler.transform(q.reshape(1, -1))[0]
        dist = np.sqrt(((Z - zq) ** 2).sum(axis=1))
    else:
        dist = np.zeros(n, dtype=float)
    k = min(_K, n)
    idx = np.argsort(dist, kind="stable")[:k]

    # Per goal: median of neighbours as the estimate, quantiles as the range.
    result: dict[str, tuple[float, float, float]] = {}
    for col in goal_cols:
        y = np.array([float(r[col]) for r in rows], dtype=float)
        neigh = y[idx]
        estimate = float(np.median(neigh))
        low = float(np.quantile(neigh, _LOW_Q))
        high = float(np.quantile(neigh, _HIGH_Q))
        result[header[col]] = (estimate, low, high)
    return result

