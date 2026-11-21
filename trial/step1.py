import csv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_cache = {}


def _num(v):
    try:
        return float(v)
    except ValueError:
        return v.strip()


def _load(path):
    if path in _cache:
        return _cache[path]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = [{k: _num(v) for k, v in r.items()} for r in reader]

    inputs, maxg, ming = [], [], []
    for c in cols:
        if c.endswith("+"):
            maxg.append(c)
        elif c.endswith("-"):
            ming.append(c)
        elif c.endswith("X"):
            continue
        else:
            inputs.append(c)

    X = np.array([[float(r[c]) for c in inputs] for r in rows], dtype=float)
    X = X.reshape(len(rows), len(inputs))
    G = np.array(
        [[float(r[c]) for c in maxg] + [-float(r[c]) for c in ming] for r in rows],
        dtype=float,
    ).reshape(len(rows), len(maxg) + len(ming))

    d = {"rows": rows, "X": X, "G": G}
    _cache[path] = d
    return d


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    d = _load(train_csv)
    X = d["X"]
    n = len(X)
    if n == 0:
        return []
    distinct = np.unique(X, axis=0)
    if len(distinct) < 3:
        return [0] * n

    best_s, best_labels = None, None
    for k in range(2, min(9, len(distinct))):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        if len(set(labels.tolist())) < 2:
            continue
        s = silhouette_score(X, labels)
        if best_s is None or s > best_s:
            best_s, best_labels = s, labels
    if best_labels is None:
        return [0] * n
    return [int(x) for x in best_labels]


def best(train_csv) -> list[int]:
    """Row indices of unusually good rows (best on all goals together), at most 5% of rows."""
    d = _load(train_csv)
    G = d["G"]
    n = len(G)
    cap = int(0.05 * n)
    if cap == 0 or G.shape[1] == 0:
        return []

    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (G - mu) / sd  # higher is better on every column

    # Pareto front: rows not dominated by any other row on all goals.
    nondominated = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(Z >= Z[i], axis=1) & np.any(Z > Z[i], axis=1)
        if dominated.any():
            nondominated[i] = False
    cand = np.flatnonzero(nondominated)

    # Rank the front by average standardized goal score, keep at most 5%.
    score = Z.mean(axis=1)
    order = cand[np.argsort(-score[cand], kind="stable")]
    return sorted(int(i) for i in order[:cap])

