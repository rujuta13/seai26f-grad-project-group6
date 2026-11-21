import csv
import math

import numpy as np


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _split_columns(cols):
    goal_max = [c for c in cols if c.endswith("+")]
    goal_min = [c for c in cols if c.endswith("-")]
    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]
    return inputs, goal_max, goal_min


def _sqdist(A, B):
    """Pairwise squared Euclidean distances between rows of A and rows of B."""
    sa = (A ** 2).sum(axis=1)
    sb = (B ** 2).sum(axis=1)
    D = sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D, 0.0)


def _robust_scale(vals):
    """Median and a robust spread (MAD, then IQR, then SD). Spread may be 0."""
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    if mad > 0:
        return med, mad
    q75, q25 = np.quantile(vals, [0.75, 0.25])
    iqr = float(q75 - q25) / 1.349
    if iqr > 0:
        return med, iqr
    return med, float(vals.std())


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv.

    Two signals, both relative to train_csv only:
      * input novelty: distance to the nearest training row in standardised
        input space, divided by the typical leave-one-out training distance;
      * goal oddity: for each goal, the robust z-score of the value against the
        training goal distribution (the largest over goals is used). A missing
        or non-finite goal cell counts as a strong glitch signal.
    The score is log1p(novelty) + log1p(goal oddity), so 0 is the most ordinary.
    """
    cols, train_rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max

    m = len(rows)
    n = len(train_rows)
    if m == 0:
        return []
    if n == 0:
        return [0.0] * m

    # ---- input novelty ----
    nov = np.zeros(m, dtype=float)
    d = len(inputs)
    if d > 0:
        X = np.array(
            [[_to_float(r.get(c)) for c in inputs] for r in train_rows],
            dtype=float,
        ).reshape(n, d)
        X = np.where(np.isfinite(X), X, 0.0)
        Xn = np.array(
            [[_to_float(r.get(c)) for c in inputs] for r in rows],
            dtype=float,
        ).reshape(m, d)
        Xn = np.where(np.isfinite(Xn), Xn, 0.0)

        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        Zn = (Xn - mu) / sd

        scale = 1.0
        if n >= 2:
            D = _sqdist(Z, Z)
            np.fill_diagonal(D, np.inf)
            loo = np.sqrt(D.min(axis=1))
            pos = loo[np.isfinite(loo) & (loo > 0)]
            if pos.size:
                scale = float(np.median(pos))

        nn = np.sqrt(_sqdist(Zn, Z).min(axis=1))
        nov = nn / scale

    # ---- goal oddity ----
    gz = np.zeros(m, dtype=float)
    for g in goals:
        tv_all = np.array([_to_float(r.get(g)) for r in train_rows], dtype=float)
        tv = tv_all[np.isfinite(tv_all)]
        if tv.size == 0:
            continue
        med, sc = _robust_scale(tv)
        for i, r in enumerate(rows):
            v = _to_float(r.get(g))
            if not math.isfinite(v):
                z = 50.0  # missing or non-finite goal: strong glitch signal
            elif sc > 0:
                z = abs(v - med) / sc
            else:
                z = 0.0 if v == med else 50.0
            if z > gz[i]:
                gz[i] = z

    score = np.log1p(nov) + np.log1p(gz)
    return [float(s) if math.isfinite(s) else 0.0 for s in score]

