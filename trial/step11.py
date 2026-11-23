"""step11.py: player types (families) and strong-and-good-from-the-spot players at full scale.

Same definitions as step1.families and step1.best, but built to stay workable on
about 17,000 rows: the silhouette score is estimated on a fixed random sample, and
the Pareto front is computed in blocks so the full pairwise comparison never has
to fit in memory at once. Loading and encoding come from step1.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from step1 import _load, _encode

# Rows used to estimate the silhouette score (exact scoring is quadratic in rows).
_SIL_SAMPLE = 2000
# Rows compared per block when finding the Pareto front (bounds memory use).
_BLOCK = 256


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    rows, inputs, _ = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    if not inputs:
        return [0] * n

    X = StandardScaler().fit_transform(_encode(rows, inputs))

    rng = np.random.default_rng(0)
    sil_idx = None
    if n > _SIL_SAMPLE:
        sil_idx = rng.choice(n, size=_SIL_SAMPLE, replace=False)

    best_labels = np.zeros(n, dtype=int)
    best_score = -np.inf
    for k in range(2, min(8, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        if len(set(labels.tolist())) < 2:
            continue
        try:
            if sil_idx is None:
                score = silhouette_score(X, labels)
            else:
                score = silhouette_score(X[sil_idx], labels[sil_idx])
        except ValueError:
            # Sampled labels collapsed to one cluster; this k cannot be scored.
            continue
        if score > best_score:
            best_score = score
            best_labels = labels

    return [int(x) for x in best_labels]


def _pareto_front(M):
    """Indices of rows not dominated by any other row (larger is better everywhere)."""
    n = M.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for s in range(0, n, _BLOCK):
        Mi = M[s:s + _BLOCK][:, None, :]
        Mj = M[None, :, :]
        ge = (Mj >= Mi).all(axis=2)   # row j is at least as good on every goal
        gt = (Mj > Mi).any(axis=2)    # row j is strictly better on some goal
        dominated[s:s + _BLOCK] = (ge & gt).any(axis=1)
    return np.flatnonzero(~dominated)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows. Uses the Pareto front over all goals, ranked when the
    front is larger than the cap."""
    rows, _, goals = _load(train_csv)
    n = len(rows)
    if n == 0 or not goals:
        return []

    # Larger is always better: minimised goals are negated.
    M = np.array(
        [[float(r[name]) * sign for name, sign in goals] for r in rows],
        dtype=float,
    )

    front_idx = _pareto_front(M)

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

