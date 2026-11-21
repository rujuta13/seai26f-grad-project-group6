import numpy as np

from step1 import _load
from step2 import _columns, _as_float
from step3 import same

_WINDOW = 20  # recent rows compared against the training reference


def _goal_layout(train_csv):
    """Goal columns in the same order step1._load uses: maximised, then minimised."""
    _, goals = _columns(train_csv)
    maxg = [g for g in goals if g.endswith("+")]
    ming = [g for g in goals if g.endswith("-")]
    return maxg, ming


def _goal_vec(row, maxg, ming):
    return ([_as_float(row.get(g, 0.0)) for g in maxg]
            + [-_as_float(row.get(g, 0.0)) for g in ming])


def stream(train_csv, rows) -> list[bool]:
    """Keep a row if no earlier row (training or streamed) dominates it on the
    standardized goals, and its average standardized score is in the top 5% of
    the reference rows. Only training data and earlier rows are used."""
    maxg, ming = _goal_layout(train_csv)
    if not (maxg or ming):
        return [False] * len(rows)

    G0 = _load(train_csv)["G"]
    if len(G0) == 0:
        return [False] * len(rows)

    mu = G0.mean(axis=0)
    sd = G0.std(axis=0)
    sd[sd == 0] = 1.0
    ref = (G0 - mu) / sd  # higher is better on every column

    decisions = []
    for row in rows:
        z = (np.array(_goal_vec(row, maxg, ming), dtype=float) - mu) / sd
        dominated = bool(np.any(np.all(ref >= z, axis=1) & np.any(ref > z, axis=1)))
        threshold = float(np.quantile(ref.mean(axis=1), 0.95))
        keep = (not dominated) and float(z.mean()) >= threshold
        decisions.append(bool(keep))
        # Every row, kept or not, becomes part of the reference for later rows.
        ref = np.vstack([ref, z.reshape(1, -1)])
    return decisions


def _nn_distances(A, B):
    """Distance from each row of A to its nearest row of B (Euclidean)."""
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(sq, 0.0)).min(axis=1)


def drift(train_csv, rows) -> int | None:
    """Warn at the first index where the last _WINDOW rows' nearest-training-row
    distances differ from the training set's own leave-one-out nearest-neighbour
    distances (permutation test from step3). Uses only rows up to that index."""
    inputs, _ = _columns(train_csv)
    X0 = _load(train_csv)["X"]
    n = len(X0)
    if n < 2 or not inputs or not rows:
        return None

    # Leave-one-out nearest-neighbour distances within the training set.
    sq0 = (X0 ** 2).sum(axis=1)
    D0 = np.sqrt(np.maximum(sq0[:, None] + sq0[None, :] - 2.0 * (X0 @ X0.T), 0.0))
    np.fill_diagonal(D0, np.inf)
    ref_nn = D0.min(axis=1)

    Xn = np.array(
        [[_as_float(r.get(c, 0.0)) for c in inputs] for r in rows], dtype=float
    ).reshape(len(rows), len(inputs))
    new_nn = _nn_distances(Xn, X0)

    for i in range(_WINDOW - 1, len(rows)):
        window = new_nn[i - _WINDOW + 1:i + 1]
        if not same(ref_nn, window):
            return i
    return None

