"""step9.py: flag benchmark rows that look like glitches or are unlike the training data.

odd() returns one suspicion score per row (higher = more suspicious). Two signals are
measured against train_csv and the larger one is kept:

* Goal glitch: a goal value that is far from what the training builds with similar
  inputs measured (robust z-score against the k nearest training rows), or far from
  every training goal (global robust z-score). Missing, non-numeric or non-finite
  goals score inf.
* Unlike anything seen: distance to the nearest training row in standardised input
  space, relative to the typical leave-one-out distance among training rows (the
  reference used by step4.drift). Unseen categorical values and unparseable numeric
  inputs score inf.

Encoding, reference distances and value coercion are reused from step1 and step4.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _load, _to_value
from step4 import _encode_row, _encode_train, _loo_nearest

_K = 5               # training neighbours used for the local goal check
_MAD_TO_SD = 1.4826  # scale factor turning a median absolute deviation into a sd-like unit


def _robust_scale(v):
    """Median and a robust spread (falls back to sd, then 1.0, if the spread is zero)."""
    med = float(np.median(v))
    mad = _MAD_TO_SD * float(np.median(np.abs(v - med)))
    if mad > 0:
        return med, mad
    sd = float(np.std(v))
    return med, (sd if sd > 0 else 1.0)


def _goal_values(row, goal_names):
    """Goal values of one row as floats, or None if any is missing or not finite."""
    out = []
    for name in goal_names:
        try:
            v = float(row.get(name))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out


def _input_unusable(row, header, inputs, specs):
    """True if an input cell is an unseen category or an unparseable/non-finite number."""
    for j, col in enumerate(inputs):
        raw = row.get(header[col])
        kind, info = specs[j]
        if kind == "num":
            v = _to_value(raw) if raw is not None else None
            if not isinstance(v, float) or not math.isfinite(v):
                return True
        else:
            if raw is None or str(raw) not in info:
                return True
    return False


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more suspicious.

    A score of inf marks a row with a missing or non-finite goal, an unparseable or
    non-finite numeric input, or a categorical input value never seen in train_csv.
    """
    if not rows:
        return []

    header, train_rows = _load(train_csv)
    maximise, minimise, inputs = _classify(header)
    goal_cols = minimise + maximise
    goal_names = [header[c] for c in goal_cols]
    m = len(goal_cols)
    n = len(train_rows)

    if n == 0:
        return [math.inf] * len(rows)

    # Training goals (raw values, so the sign of maximised goals does not matter here).
    Y = np.array([[float(r[c]) for c in goal_cols] for r in train_rows],
                 dtype=float).reshape(n, m)
    gscale = [_robust_scale(Y[:, j]) for j in range(m)]

    # Training inputs: encoded, standardised, and the leave-one-out reference distance.
    X, specs = _encode_train(train_rows, inputs)
    if inputs:
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
    else:
        scaler = None
        Z = np.zeros((n, 0), dtype=float)
    ref = np.asarray(_loo_nearest(Z), dtype=float)
    typical = float(np.median(ref)) if n >= 2 else 1.0
    if not (0.0 < typical < math.inf):
        typical = 1.0

    scores: list[float] = []
    for row in rows:
        g = _goal_values(row, goal_names)
        if g is None or (inputs and _input_unusable(row, header, inputs, specs)):
            scores.append(math.inf)
            continue

        # Input novelty: distance to the nearest training row, in typical-spread units.
        if inputs:
            q = _encode_row(row, header, inputs, specs)
            zq = scaler.transform(q.reshape(1, -1))[0]
            dist = np.sqrt(((Z - zq) ** 2).sum(axis=1))
        else:
            dist = np.zeros(n, dtype=float)
        novelty = float(dist.min()) / typical

        # Goal glitch: compare each goal with the global training spread and with the
        # goals of the nearest training rows (same flags, different number?).
        k = min(_K, n)
        idx = np.argsort(dist, kind="stable")[:k]
        goal_score = 0.0
        gv = np.asarray(g, dtype=float)
        for j in range(m):
            med, scale = gscale[j]
            glob = abs(gv[j] - med) / scale
            local_med = float(np.median(Y[idx, j]))
            loc = abs(gv[j] - local_med) / scale
            goal_score = max(goal_score, glob, loc)

        scores.append(float(max(novelty, goal_score)))

    return scores

