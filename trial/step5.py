"""step5.py: name the input columns most to blame for bad rows.

A row is "bad" when its mean goal rank (over all goals, worst-first) falls in the
top quarter of the data. Each input column is then scored by the mutual information
between its values and that bad/good label, so columns that separate bad rows from
good ones rank first. Reuses the loading, classification and goal helpers from step1.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from step1 import _classify, _goal_matrix, _input_matrix, _load

_BAD_QUANTILE = 0.75   # rows above this quantile of mean goal rank count as bad
_DISCRETE_MAX = 20     # columns with at most this many distinct values are treated as discrete
_SEED = 0              # fixed seed so repeated calls give the same ranking


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    header, rows = _load(train_csv)
    maximise, minimise, inputs = _classify(header)
    names = [header[i] for i in inputs]
    n = len(rows)
    if not inputs or n < 2:
        return names

    G = _goal_matrix(rows, maximise, minimise)
    if G.shape[1] == 0:
        return names

    # Mean rank across goals: higher rank means worse (goals are minimised in G).
    ranks = np.empty_like(G)
    for k in range(G.shape[1]):
        order = np.argsort(G[:, k], kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        ranks[:, k] = r
    score = ranks.mean(axis=1)

    thr = np.quantile(score, _BAD_QUANTILE)
    bad = score > thr
    if not bad.any() or bad.all():
        bad = score >= thr
    if not bad.any() or bad.all():
        # Goals do not separate the rows at all, so no column can be blamed.
        return names

    y = bad.astype(int)
    X = _input_matrix(rows, inputs)
    discrete = np.array(
        [len(np.unique(X[:, j])) <= _DISCRETE_MAX for j in range(X.shape[1])],
        dtype=bool,
    )
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=_SEED)

    order = sorted(range(len(names)), key=lambda j: (-float(mi[j]), j))
    return [names[j] for j in order]

