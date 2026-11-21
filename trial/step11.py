"""step11.py: player types and strong-and-good-from-the-spot players at scale.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers,
input encoding and scaling, k-means, silhouette, goal scoring and the
5% cap). Only the scalable parts are new.

families(): step1.families runs k-means on every row. That is too slow in
pure Python for about 17,000 rows, so for larger files k and the clusters
are chosen on a fixed random sample of _FIT_CAP rows (same k range and
silhouette rule as step1). Every row is then assigned to its nearest
cluster centre. Files of _FIT_CAP rows or fewer use step1.families as is.

best(): the same Pareto front and the same 5% cap as step1.best, with the
same ordering. step1 compares every pair of rows, which is O(n^2). Here
rows are visited in decreasing order of their goal-score sum. A row can
only be dominated by a row with a strictly larger sum, and dominance is
transitive, so a row is non-dominated exactly when no row already on the
front dominates it.
"""
from __future__ import annotations

import math
import os
import random

from step1 import families as _step1_families
from step1 import (
    _BEST_DEN,
    _BEST_NUM,
    _MAX_K,
    _SEED,
    _best_kmeans,
    _classify,
    _encode_inputs,
    _goal_scores,
    _minmax,
    _read,
    _relabel,
    _silhouette,
)

_FIT_CAP = 1000  # rows used to choose k and fit the clusters on large files


def _centers(X: list[list[float]], labels: list[int]) -> dict[int, list[float]]:
    """Mean vector of each label that is used."""
    dims = len(X[0])
    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for x, c in zip(X, labels):
        if c not in sums:
            sums[c] = [0.0] * dims
            counts[c] = 0
        s = sums[c]
        for d in range(dims):
            s[d] += x[d]
        counts[c] += 1
    return {c: [v / counts[c] for v in s] for c, s in sums.items()}


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Labels are contiguous integers starting at 0. For files larger than
    _FIT_CAP rows, k and the cluster centres come from a random sample and
    all rows are then assigned to the nearest centre.
    """
    path = os.fspath(train_csv)
    header, rows = _read(path)
    inputs, _ = _classify(header)
    n = len(rows)
    if n == 0:
        return []
    if not inputs:
        return [0] * n
    if n <= _FIT_CAP:
        return _step1_families(path)

    X = _minmax(_encode_inputs(rows, inputs))
    rng = random.Random(_SEED)
    fit_idx = sorted(rng.sample(range(n), _FIT_CAP))
    Xf = [X[i] for i in fit_idx]

    chosen: list[int] | None = None
    chosen_score = -math.inf
    for k in range(2, min(_MAX_K, len(Xf)) + 1):
        labels = _best_kmeans(Xf, k, rng)
        score = _silhouette(Xf, labels, rng)
        if score > chosen_score:
            chosen, chosen_score = labels, score
    if chosen is None:
        return [0] * n

    centers = _centers(Xf, chosen)
    keys = list(centers)
    cens = [centers[c] for c in keys]
    out: list[int] = []
    for x in X:
        best_j = 0
        best_d = math.inf
        for j, cen in enumerate(cens):
            d = math.dist(x, cen)
            if d < best_d:
                best_j, best_d = j, d
        out.append(keys[best_j])
    return _relabel(out)


def _dominates(f: list[float], s: list[float]) -> bool:
    """True if f is at least as good as s on every goal and better on one."""
    strict = False
    for a, b in zip(f, s):
        if a < b:
            return False
        if a > b:
            strict = True
    return strict


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together).

    The Pareto front over the normalised goals (1 is best for each goal),
    ordered by mean score. At most 5% of the rows are returned.
    """
    header, rows = _read(os.fspath(train_csv))
    _, goals = _classify(header)
    n = len(rows)
    cap = n * _BEST_NUM // _BEST_DEN
    if n == 0 or not goals or cap == 0:
        return []

    S = _goal_scores(rows, goals)
    sums = [sum(s) for s in S]
    order = sorted(range(n), key=lambda i: (-sums[i], i))

    front: list[int] = []
    for i in order:
        si = S[i]
        if any(_dominates(S[f], si) for f in front):
            continue
        front.append(i)

    front.sort(key=lambda i: (-sums[i], i))
    return front[:cap]

