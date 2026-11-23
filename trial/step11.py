import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from step1 import _load
from step2 import _columns

# Columns that identify a player or record rather than describe them.
_IDENTIFIERS = {"num", "Rank", "Name", "url"}
# "Strong and good from the spot": Strength and Penalties, both maximised.
_SPOT_GOALS = ("Strength+", "Penalties+")


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Clusters on the player attributes only: identifier columns are left out,
    and the attributes are standardized so no single scale dominates.
    """
    inputs, _ = _columns(train_csv)
    rows = _load(train_csv)["rows"]
    n = len(rows)
    if n == 0:
        return []

    keep = [c for c in inputs if c not in _IDENTIFIERS] or list(inputs)
    if not keep:
        return [0] * n

    X = np.array([[float(r[c]) for c in keep] for r in rows], dtype=float)
    X = X.reshape(n, len(keep))
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    distinct = np.unique(X, axis=0)
    if len(distinct) < 3:
        return [0] * n

    best_s, best_labels = None, None
    for k in range(2, min(9, len(distinct))):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        if len(set(labels.tolist())) < 2:
            continue
        # Silhouette on a fixed sample keeps this fast for ~17,000 players.
        s = silhouette_score(
            X, labels,
            sample_size=min(n, 3000),
            random_state=0,
        )
        if best_s is None or s > best_s:
            best_s, best_labels = s, labels
    if best_labels is None:
        return [0] * n
    return [int(x) for x in best_labels]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows on Strength and Penalties together.

    A row qualifies if no other row is at least as good on both goals and
    strictly better on one. Qualifying rows are ranked by their average
    standardized score on these two goals; at most 5% of rows are returned.
    """
    rows = _load(train_csv)["rows"]
    n = len(rows)
    cap = int(0.05 * n)
    if cap == 0 or not rows:
        return []
    if any(c not in rows[0] for c in _SPOT_GOALS):
        return []

    G = np.array([[float(r[c]) for c in _SPOT_GOALS] for r in rows], dtype=float)
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (G - mu) / sd  # higher is better on both goals

    # Pareto front: rows not dominated by any other row on both goals.
    nondominated = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(Z >= Z[i], axis=1) & np.any(Z > Z[i], axis=1)
        if dominated.any():
            nondominated[i] = False
    cand = np.flatnonzero(nondominated)

    score = Z.mean(axis=1)
    order = cand[np.argsort(-score[cand], kind="stable")]
    return sorted(int(i) for i in order[:cap])

