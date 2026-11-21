"""step1.py: flag-combination families (clusters) and unusually good rows.

No earlier step modules were provided with this request, so the CSV loading and
encoding helpers needed here are defined in this file instead of being imported.
"""

import csv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _load(path):
    """Return (rows, input_cols, goals) where goals is a list of (name, sign)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        names = reader.fieldnames or []
        rows = list(reader)
    inputs, goals = [], []
    for name in names:
        if name.endswith("+"):
            goals.append((name, 1.0))
        elif name.endswith("-"):
            goals.append((name, -1.0))
        elif name.endswith("X"):
            continue
        else:
            inputs.append(name)
    return rows, inputs, goals


def _encode(rows, cols):
    """Numeric matrix for input columns; non-numeric columns get ordinal codes."""
    out = np.zeros((len(rows), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        vals = [r[c] for r in rows]
        try:
            out[:, j] = [float(v) for v in vals]
        except (TypeError, ValueError):
            codes = {v: i for i, v in enumerate(sorted(set(map(str, vals))))}
            out[:, j] = [codes[str(v)] for v in vals]
    return out


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    rows, inputs, _ = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    if not inputs:
        return [0] * n

    X = StandardScaler().fit_transform(_encode(rows, inputs))

    best_labels = np.zeros(n, dtype=int)
    best_score = -np.inf
    for k in range(2, min(8, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        if len(set(labels.tolist())) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    return [int(x) for x in best_labels]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows. Uses the Pareto front over all goals, ranked when the
    front is larger than the cap."""
    rows, _, goals = _load(train_csv)
    n = len(rows)
    if n == 0 or not goals:
        return []

    # Build a matrix where larger is always better.
    M = np.array(
        [[float(r[name]) * sign for name, sign in goals] for r in rows],
        dtype=float,
    )

    # Non-dominated (Pareto) front. ge[i, j]: row j >= row i on all goals.
    # gt[i, j]: row j > row i on at least one goal. Row i is dominated if some j does both.
    M_j = M[None, :, :]
    M_i = M[:, None, :]
    ge = (M_j >= M_i).all(axis=2)
    gt = (M_j > M_i).any(axis=2)
    dominated = (ge & gt).any(axis=1)
    front_idx = np.flatnonzero(~dominated)

    cap = max(1, int(0.05 * n))
    if len(front_idx) <= cap:
        return sorted(int(i) for i in front_idx)

    # Too many non-dominated rows: keep the ones with the best normalised total.
    F = M[front_idx]
    mins = F.min(axis=0)
    maxs = F.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    score = ((F - mins) / span).sum(axis=1)
    order = np.argsort(-score, kind="stable")
    chosen = front_idx[order[:cap]]
    return sorted(int(i) for i in chosen)

