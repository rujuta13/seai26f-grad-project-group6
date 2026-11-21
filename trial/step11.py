"""step11.py - client question 11: remember.

families(train_csv) sorts the flag sets (players) into types. step1.families
builds an n-by-n distance matrix, which cannot hold 17,000 rows, so this
version repeats the same steps on a sample. k-means and the silhouette score
choose the number of families on at most SAMPLE_SIZE rows. Each row then gets
the family of its nearest centroid. Inputs are standardised as step2 does, so
columns with large numbers (such as ranks) do not dominate the distances.

best(train_csv) returns the unusually good rows: those not dominated on all
goals together, at most 5% of rows. step1.best already does this, so it is
reused.

Reused from earlier steps:
  step1: SEED, MIN_FAMILIES, MAX_FAMILIES, _best_kmeans, _silhouette, _sqdist,
         best
  step2: _model (standardised inputs)
"""
import math
import random

from step1 import (
    MAX_FAMILIES,
    MIN_FAMILIES,
    SEED,
    _best_kmeans,
    _silhouette,
    _sqdist,
    best as _best_rows,
)
from step2 import _model

SAMPLE_SIZE = 1000


def _centres(X, labels):
    """Centroid of each non-empty cluster, in label order."""
    k = max(labels) + 1
    d = len(X[0])
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    for x, lab in zip(X, labels):
        counts[lab] += 1
        acc = sums[lab]
        for t in range(d):
            acc[t] += x[t]
    return [[v / counts[c] for v in sums[c]] for c in range(k) if counts[c]]


def _nearest(z, centres):
    """Position of the centroid nearest to z."""
    return min(range(len(centres)), key=lambda c: _sqdist(z, centres[c]))


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    The number of families is chosen from MIN_FAMILIES..MAX_FAMILIES by
    silhouette score, computed on a sample of at most SAMPLE_SIZE rows.
    """
    _, _, _, _, Z, _ = _model(train_csv)
    n = len(Z)
    if n < 3:
        return [0] * n

    rng = random.Random(SEED)
    if n > SAMPLE_SIZE:
        idx = sorted(rng.sample(range(n), SAMPLE_SIZE))
    else:
        idx = list(range(n))
    S = [Z[i] for i in idx]
    s = len(S)

    D = [[0.0] * s for _ in range(s)]
    for i in range(s):
        for j in range(i + 1, s):
            D[i][j] = D[j][i] = math.sqrt(_sqdist(S[i], S[j]))

    best_centres, best_score = None, -math.inf
    for k in range(MIN_FAMILIES, min(MAX_FAMILIES, s - 1) + 1):
        labels = _best_kmeans(S, k, rng)
        score = _silhouette(D, labels)
        if score > best_score:
            best_score = score
            best_centres = _centres(S, labels)

    if not best_centres:
        return [0] * n
    return [_nearest(z, best_centres) for z in Z]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows, in ascending order."""
    return _best_rows(train_csv)

