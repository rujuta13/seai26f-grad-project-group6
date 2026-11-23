"""step12.py: guess a player's goals from his other numbers, with a range.

Reuses step2's neighbour settings and step4's encoding helpers. The one new
choice is that identifier columns (num, Name, url) are left out of the
neighbour search, as step11 does for clustering. Their values are row labels,
so they would otherwise pull the neighbours toward players who happen to be
listed near each other.

The range low..high is the 10th to 90th percentile of the goal values among
the nearest training players. A wide range means the nearest players disagree,
so the guess is less certain.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _load
from step2 import _HIGH_Q, _K, _LOW_Q
from step4 import _encode_row, _encode_train
from step11 import _trait_inputs


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of `row` as (estimate, low, high).

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns {goal column name: (estimate, low, high)} for every goal in the
    training file, in header order. The range low..high is the span the true
    value is expected to fall in, taken from the nearest training players.
    """
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")

    _, _, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)
    goal_cols = [i for i, name in enumerate(header) if name.endswith(("+", "-"))]
    n = len(rows)

    if inputs:
        X, specs = _encode_train(rows, inputs)
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        q = _encode_row(row, header, inputs, specs)
        zq = scaler.transform(q.reshape(1, -1))[0]
        dist = np.sqrt(((Z - zq) ** 2).sum(axis=1))
    else:
        dist = np.zeros(n, dtype=float)

    k = min(_K, n)
    idx = np.argsort(dist, kind="stable")[:k]

    result: dict[str, tuple[float, float, float]] = {}
    for col in goal_cols:
        y = np.array([float(r[col]) for r in rows], dtype=float)
        neigh = y[idx]
        estimate = float(np.median(neigh))
        low = float(np.quantile(neigh, _LOW_Q))
        high = float(np.quantile(neigh, _HIGH_Q))
        result[header[col]] = (estimate, low, high)
    return result

