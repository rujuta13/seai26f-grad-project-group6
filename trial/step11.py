"""step11.py: player families and the strong-and-good-from-the-spot shortlist.

families() reuses step1's loading, classification, input encoding and KMeans and
silhouette search, with one change for this file: identifier columns (num, Name,
url) are left out of the clustering. Their values are row labels, not player
traits, and category-coding them would split the players by name order.

best() is step1.best(): the Pareto front on all goals (here Strength+ and
Penalties+), capped at 5% of rows, so "strong and good from the spot" is the
front of those two goals.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import step1
from step1 import _classify, _input_matrix, _load

_KMAX = 8                     # largest number of families considered
_SEED = 0                     # fixed seed so repeated calls give the same answer
_SILHOUETTE_SAMPLE = 3000     # silhouette is computed on a fixed random subsample above this size
_ID_COLUMNS = {"num", "name", "url"}   # identifiers, not player traits


def _trait_inputs(header, inputs):
    """Input column indices with identifier columns removed."""
    return [c for c in inputs if header[c].strip().lower() not in _ID_COLUMNS]


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    header, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    _, _, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)
    if not inputs:
        return [0] * n

    X = StandardScaler().fit_transform(_input_matrix(rows, inputs))
    distinct = len(np.unique(X, axis=0))
    if distinct < 2:
        return [0] * n

    rng = np.random.default_rng(_SEED)
    if n > _SILHOUETTE_SAMPLE:
        sample = np.sort(rng.choice(n, size=_SILHOUETTE_SAMPLE, replace=False))
    else:
        sample = np.arange(n)

    best_labels, best_score = None, -math.inf
    k_max = min(_KMAX, distinct - 1, n - 1)
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=_SEED)
        labels = km.fit_predict(X)
        if len(set(labels.tolist())) < 2:
            continue
        try:
            score = silhouette_score(X[sample], labels[sample])
        except ValueError:
            # The subsample can miss a cluster entirely; skip this k.
            continue
        if score > best_score:
            best_score, best_labels = score, labels

    if best_labels is None:
        # Too few distinct points for the search range; fall back to two clusters.
        best_labels = KMeans(n_clusters=2, n_init=10, random_state=_SEED).fit_predict(X)

    return [int(v) for v in best_labels]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together), at most 5% of rows.

    Delegates to step1.best: the Pareto front on all goals, ordered by mean goal rank
    and capped at 5% of the data. Returned indices are sorted.
    """
    return step1.best(train_csv)

