"""step1.py: flag-combination families and best-row detection for the MOOT data.

Note: no earlier step modules were available to import from, so this file is
self-contained. It uses only the standard library, numpy 1.26 and scikit-learn 1.7.
"""
from __future__ import annotations

import csv
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_BUDGET_FRACTION = 0.05   # "best" rows may be at most 5% of the data
_KMAX = 8                 # largest number of families considered
_SEED = 0
_CHUNK = 512              # chunk size for the O(n^2) dominance test


def _load(path):
    """Return (header, rows) where rows are lists of raw string cells."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def _to_value(cell):
    try:
        return float(cell)
    except ValueError:
        return cell


def _classify(header):
    """Split column indices into (maximise, minimise, inputs). 'X' columns are dropped."""
    maximise, minimise, inputs = [], [], []
    for i, name in enumerate(header):
        if name.endswith("+"):
            maximise.append(i)
        elif name.endswith("-"):
            minimise.append(i)
        elif name.endswith("X"):
            continue
        else:
            inputs.append(i)
    return maximise, minimise, inputs


def _input_matrix(rows, inputs):
    """Numeric matrix of the input columns; non-numeric columns are category-coded."""
    n = len(rows)
    X = np.zeros((n, len(inputs)), dtype=float)
    for j, col in enumerate(inputs):
        vals = [_to_value(r[col]) for r in rows]
        if all(isinstance(v, float) for v in vals):
            X[:, j] = vals
        else:
            codes = {v: k for k, v in enumerate(sorted({str(v) for v in vals}))}
            X[:, j] = [codes[str(v)] for v in vals]
    return X


def _goal_matrix(rows, maximise, minimise):
    """Goals as a minimisation matrix: maximised columns are negated."""
    cols = []
    for col in minimise:
        cols.append([float(r[col]) for r in rows])
    for col in maximise:
        cols.append([-float(r[col]) for r in rows])
    if not cols:
        return np.zeros((len(rows), 0), dtype=float)
    return np.array(cols, dtype=float).T


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    header, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    _, _, inputs = _classify(header)
    if not inputs:
        return [0] * n

    X = StandardScaler().fit_transform(_input_matrix(rows, inputs))
    distinct = len(np.unique(X, axis=0))
    if distinct < 2:
        return [0] * n

    # Choose k in [2, _KMAX] by silhouette score.
    best_labels, best_score = None, -math.inf
    k_max = min(_KMAX, distinct - 1, n - 1)
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=_SEED)
        labels = km.fit_predict(X)
        if len(set(labels.tolist())) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_labels = score, labels

    if best_labels is None:
        # Too few distinct points for the search range; fall back to two clusters.
        best_labels = KMeans(n_clusters=2, n_init=10, random_state=_SEED).fit_predict(X)

    return [int(v) for v in best_labels]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together).

    A row is kept if no other row is at least as good on every goal and strictly
    better on one (the Pareto front). If the front holds more rows than 5% of the
    data, the rows with the best mean goal rank are kept. Returned indices are sorted.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    maximise, minimise, _ = _classify(header)
    G = _goal_matrix(rows, maximise, minimise)
    if n == 0 or G.shape[1] == 0:
        return []

    # Pareto front (all goals are minimised in G). Chunked to bound memory.
    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        Gi = G[start:stop, None, :]   # (c, 1, m)
        Gj = G[None, :, :]            # (1, n, m)
        le = (Gj <= Gi).all(axis=2)   # row j is no worse than row i on every goal
        lt = (Gj < Gi).any(axis=2)    # row j is strictly better on some goal
        dominated[start:stop] = (le & lt).any(axis=1)
    front = np.flatnonzero(~dominated)

    # Mean rank across goals (lower is better), used to order and cap the front.
    ranks = np.empty_like(G)
    for k in range(G.shape[1]):
        order = np.argsort(G[:, k], kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        ranks[:, k] = r
    score = ranks.mean(axis=1)

    cap = int(math.floor(_BUDGET_FRACTION * n))
    if cap < 1:
        return []
    ordered = front[np.argsort(score[front], kind="stable")]
    chosen = ordered[:cap]
    return sorted(int(i) for i in chosen)

