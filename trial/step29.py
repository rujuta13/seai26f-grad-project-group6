import math

import numpy as np

from step1 import _load
from step2 import _columns
from step4 import _goal_layout, _nn_distances
from step9 import _cell

_BAD_WEIGHT = 10.0    # a typo, impossible or non-numeric flag value
_MISSING_GOAL = 1.0   # one missing goal cell in an otherwise measured record
_Z_FREE = 3.0         # robust per-column z-score below this is ordinary
_CAT_MAX = 10         # columns with at most this many distinct training values are categorical


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more likely a
    glitch, a typo, or unlike anything in train_csv.

    Parts added together:
      * bad cells: non-numeric or missing flags, typos (values never seen in a
        categorical column), and impossible values (negative where training has
        none) are each weighted heavily;
      * per-column outliers: robust z-score (median and MAD) of each input against
        the training data, counting only the part beyond _Z_FREE, so an age of 150
        or an income far outside the training range stands out;
      * flag novelty: distance to the nearest training row on standardized inputs,
        in units of the typical training nearest-neighbour distance;
      * goal anomaly: the largest absolute z-score of the row's goals, with
        minimised goals sign-flipped as in step1. Missing goal cells add a small
        penalty; a record with no goals at all is not judged on goals.

    A high score flags a record for checking. It does not establish fraud.
    """
    rows = list(rows)
    if not rows:
        return []

    inputs, _ = _columns(train_csv)
    maxg, ming = _goal_layout(train_csv)

    d = _load(train_csv)
    X0, G0 = d["X"], d["G"]
    n, m, k = len(X0), len(inputs), len(rows)
    if n == 0:
        return [0.0] * k

    Xn = np.array(
        [[_cell(r.get(c, math.nan)) for c in inputs] for r in rows], dtype=float
    ).reshape(k, m)

    bad = np.sum(~np.isfinite(Xn), axis=1).astype(float)
    score = np.zeros(k)

    # Per-column checks: typos, impossible values, and robust outliers.
    for j in range(m):
        col = X0[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        x = Xn[:, j]
        ok = np.isfinite(x)

        seen = set(col.tolist())
        if len(seen) <= _CAT_MAX:
            for i in np.flatnonzero(ok):
                if float(x[i]) not in seen:
                    bad[i] += 1

        if float(col.min()) >= 0.0:
            bad[ok & (x < 0.0)] += 1

        med = float(np.median(col))
        mad = 1.4826 * float(np.median(np.abs(col - med)))
        if not mad > 0:
            mad = float(col.std()) or 1.0
        z = np.abs(np.where(ok, x, med) - med) / mad
        score += np.where(ok, np.maximum(z - _Z_FREE, 0.0), 0.0)

    # Flag novelty on standardized inputs, so large-scale columns such as
    # INCOME do not dominate the distances.
    if n >= 2 and m > 0:
        X0c = np.where(np.isfinite(X0), X0, 0.0)
        Xc = np.where(np.isfinite(Xn), Xn, 0.0)
        mu = X0c.mean(axis=0)
        sd = X0c.std(axis=0)
        sd[sd == 0] = 1.0
        Z0 = (X0c - mu) / sd
        Zn = (Xc - mu) / sd

        sq0 = (Z0 ** 2).sum(axis=1)
        D0 = np.sqrt(np.maximum(sq0[:, None] + sq0[None, :] - 2.0 * (Z0 @ Z0.T), 0.0))
        np.fill_diagonal(D0, np.inf)
        ref = D0.min(axis=1)
        pos = ref[np.isfinite(ref) & (ref > 0)]
        scale = float(np.median(pos)) if pos.size else 1.0
        if not (scale > 0 and math.isfinite(scale)):
            scale = 1.0
        score += _nn_distances(Zn, Z0) / scale

    # Goal anomaly against the training goals.
    if G0.shape[1] > 0:
        mu_g = G0.mean(axis=0)
        sd_g = G0.std(axis=0)
        sd_g[sd_g == 0] = 1.0
        for i, r in enumerate(rows):
            vec = np.array(
                [_cell(r.get(g, math.nan)) for g in maxg]
                + [-_cell(r.get(g, math.nan)) for g in ming],
                dtype=float,
            )
            fin = np.isfinite(vec)
            if not fin.any():
                continue
            score[i] += _MISSING_GOAL * float(np.sum(~fin))
            z = (vec[fin] - mu_g[fin]) / sd_g[fin]
            score[i] += float(np.max(np.abs(z)))

    score += _BAD_WEIGHT * bad
    return [float(s) for s in score]

