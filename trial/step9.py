import math

import numpy as np

from step1 import _load
from step2 import _columns
from step4 import _goal_layout, _nn_distances

_BAD_WEIGHT = 10.0  # each invalid or never-seen cell adds this much suspicion


def _cell(v):
    """A cell as a float, or NaN if it is missing or not numeric."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row (higher = more likely a glitch, typo, or unlike training).

    Three parts are added together:
      * flag novelty: distance from the row's flags to the nearest training row,
        in units of the typical training nearest-neighbour distance;
      * goal anomaly: the largest absolute z-score of the row's goals against
        the training goals (minimised goals already sign-flipped, as in step1);
      * bad cells: missing or non-numeric values, and flag values never seen in
        that column of the training data (typos), each weighted heavily.
    """
    rows = list(rows)
    if not rows:
        return []

    inputs, _ = _columns(train_csv)
    maxg, ming = _goal_layout(train_csv)
    gnames = maxg + ming

    d = _load(train_csv)
    X0 = d["X"]
    G0 = d["G"]
    n = len(X0)
    m = len(inputs)
    k = len(rows)

    # Flags of the new rows.
    Xn = np.array(
        [[_cell(r.get(c, math.nan)) for c in inputs] for r in rows], dtype=float
    ).reshape(k, m)

    bad = np.zeros(k)
    bad += np.sum(~np.isfinite(Xn), axis=1)

    # Flag values never seen in that column of the training data (typos).
    if n > 0:
        for j in range(m):
            seen = set(X0[:, j].tolist())
            for i in range(k):
                v = Xn[i, j]
                if np.isfinite(v) and v not in seen:
                    bad[i] += 1

    # Flag novelty: nearest-training-row distance, scaled by the typical
    # leave-one-out nearest-neighbour distance within the training set.
    nov = np.zeros(k)
    if n >= 2 and m > 0:
        X0c = np.where(np.isfinite(X0), X0, 0.0)
        Xc = np.where(np.isfinite(Xn), Xn, 0.0)

        sq0 = (X0c ** 2).sum(axis=1)
        D0 = np.sqrt(np.maximum(sq0[:, None] + sq0[None, :] - 2.0 * (X0c @ X0c.T), 0.0))
        np.fill_diagonal(D0, np.inf)
        ref = D0.min(axis=1)

        pos = ref[np.isfinite(ref) & (ref > 0)]
        scale = float(np.median(pos)) if pos.size else 1.0
        if not (scale > 0 and math.isfinite(scale)):
            scale = 1.0

        nov = _nn_distances(Xc, X0c) / scale

    # Goal anomaly: largest absolute z-score against the training goals.
    gz = np.zeros(k)
    if n > 0 and G0.shape[1] > 0 and gnames:
        mu = G0.mean(axis=0)
        sd = G0.std(axis=0)
        sd[sd == 0] = 1.0
        for i, r in enumerate(rows):
            vals = ([_cell(r.get(g, math.nan)) for g in maxg]
                    + [-_cell(r.get(g, math.nan)) for g in ming])
            vec = np.array(vals, dtype=float)
            finite = np.isfinite(vec)
            bad[i] += int(np.sum(~finite))
            if finite.any():
                z = (vec[finite] - mu[finite]) / sd[finite]
                gz[i] = float(np.max(np.abs(z)))

    score = nov + gz + _BAD_WEIGHT * bad
    return [float(s) for s in score]

