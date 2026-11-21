import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from step1 import _read, _columns, _to_float_column
from step2 import _encode_inputs

_K = 5
_MISSING_PENALTY = 3.0
_UNSEEN_PENALTY = 2.0
_SEED = 0
_MIN_ISO = 8


def _is_numeric(values):
    try:
        for v in values:
            float(v)
    except (TypeError, ValueError):
        return False
    return True


def _float_or_nan(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _robust_center_scale(vals):
    """Median and MAD-based scale over the finite values, with safe fallbacks."""
    vals = np.asarray(vals, dtype=float)
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return 0.0, 1.0
    med = float(np.median(fin))
    scale = 1.4826 * float(np.median(np.abs(fin - med)))
    if not scale > 0:
        scale = float(fin.std())
    if not scale > 0:
        scale = 1.0
    return med, scale


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row: higher means more likely a glitch or
    a value unlike anything in the training data.

    Components (all added together, each non-negative):
      - input-space isolation: an IsolationForest on standardised inputs,
        scored robustly against the training rows' own scores;
      - input-space novelty: mean distance to the nearest training rows,
        scored robustly against the same statistic for training rows;
      - goal outliers: for every goal, log(1 + robust |z|) against the
        training median and MAD, with a fixed penalty for missing or
        non-numeric goal values;
      - unseen categories: a fixed penalty per categorical input whose
        value never appears in the training data.
    """
    fieldnames, train_rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    m = len(rows)
    if m == 0:
        return []
    n = len(train_rows)
    if n == 0:
        return [0.0] * m

    score = np.zeros(m, dtype=float)

    # 1. Input-space features: isolation and novelty.
    if inputs:
        X, encode_row = _encode_inputs(train_rows, inputs)
        Xf = np.array(X, dtype=float, copy=True)
        for j in range(Xf.shape[1]):
            col = Xf[:, j]
            fin = np.isfinite(col)
            fill = float(col[fin].mean()) if np.any(fin) else 0.0
            Xf[~fin, j] = fill

        R = np.vstack([encode_row(r) for r in rows]).astype(float)
        R = np.where(np.isfinite(R), R, 0.0)

        mu = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Ztr = (Xf - mu) / sd
        Zr = (R - mu) / sd
    else:
        Ztr = np.zeros((n, 1))
        Zr = np.zeros((m, 1))

    # Novelty: distance to nearest training rows.
    if n >= 2:
        k_tr = min(_K + 1, n)
        nn_tr = NearestNeighbors(n_neighbors=k_tr).fit(Ztr)
        d_tr, _ = nn_tr.kneighbors(Ztr)
        ref = d_tr[:, 1:].mean(axis=1)  # drop self-match
        med, scale = _robust_center_scale(ref)

        k_r = min(_K, n)
        nn_r = NearestNeighbors(n_neighbors=k_r).fit(Ztr)
        d_r, _ = nn_r.kneighbors(Zr)
        d_r = d_r.mean(axis=1)
        score += np.maximum(0.0, (d_r - med) / scale)

    # Isolation: unusual combinations of flags.
    if n >= _MIN_ISO:
        iso = IsolationForest(random_state=_SEED).fit(Ztr)
        tr_anom = -iso.score_samples(Ztr)
        r_anom = -iso.score_samples(Zr)
        med, scale = _robust_center_scale(tr_anom)
        score += np.maximum(0.0, (r_anom - med) / scale)

    # 2. Unseen categorical values.
    if inputs:
        unseen = np.zeros(m, dtype=float)
        for c in inputs:
            tv = [r.get(c) for r in train_rows]
            if _is_numeric(tv):
                continue
            seen = set(tv)
            for i, r in enumerate(rows):
                if r.get(c) not in seen:
                    unseen[i] += 1.0
        score += _UNSEEN_PENALTY * unseen

    # 3. Goal outliers and missing goal values.
    for name, _sign in goals:
        tcol = _to_float_column(train_rows, name)
        med, scale = _robust_center_scale(tcol)
        for i, r in enumerate(rows):
            v = _float_or_nan(r.get(name))
            if not np.isfinite(v):
                score[i] += _MISSING_PENALTY
            else:
                score[i] += float(np.log1p(abs(v - med) / scale))

    return [float(x) for x in score]

