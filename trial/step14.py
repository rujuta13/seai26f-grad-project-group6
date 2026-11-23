import math

import numpy as np

from step1 import _read, _columns, _to_float_column
from step4 import stream as _stream

_ALPHA = 0.05
_FIRST_CHECK = 16


def stream(train_csv, rows) -> list[bool]:
    """Keep a streamed row iff no training row and no earlier kept row beats it on every goal."""
    return _stream(train_csv, rows)


def _z_pvalue(x, y):
    """Two-sided normal-approximation p-value for the difference in means."""
    n, m = len(x), len(y)
    if n < 2 or m < 1:
        return 1.0
    d = float(y.mean()) - float(x.mean())
    if d == 0.0:
        return 1.0
    var_x = float(x.var(ddof=1))
    var_y = float(y.var(ddof=1)) if m > 1 else 0.0
    v = var_x / n + var_y / m
    if not v > 0.0:
        return 0.0
    z = abs(d) / math.sqrt(v)
    return math.erfc(z / math.sqrt(2.0))


def _num_getter(c):
    def get(row):
        try:
            v = float(row.get(c))
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None
    return get


def _eq_getter(c, v):
    def get(row):
        return 1.0 if row.get(c) == v else 0.0
    return get


def _unseen_getter(c, seen):
    def get(row):
        return 0.0 if row.get(c) in seen else 1.0
    return get


def _build(train_csv):
    """Feature specs: (key, training values, getter mapping a row to a float or None)."""
    fieldnames, train_rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    specs = []

    for name, _sign in goals:
        col = _to_float_column(train_rows, name)
        specs.append((("goal", name), col[np.isfinite(col)], _num_getter(name)))

    for c in inputs:
        raw = [r.get(c) for r in train_rows]
        try:
            col = np.array([float(v) for v in raw], dtype=float)
        except (TypeError, ValueError):
            levels = list(dict.fromkeys(raw))
            for v in levels:
                arr = np.array([1.0 if r == v else 0.0 for r in raw], dtype=float)
                specs.append((("lvl", c, v), arr, _eq_getter(c, v)))
            seen = set(levels)
            specs.append((("unseen", c), np.zeros(len(raw), dtype=float),
                          _unseen_getter(c, seen)))
        else:
            col = col[np.isfinite(col)]
            specs.append((("in", c), col, _num_getter(c)))

    return [s for s in specs if len(s[1]) >= 2]


def drift(train_csv, rows) -> int | None:
    """Index of the first row at which new rows no longer look like train_csv, or None.

    Compares running windows of goal values and input features (numeric values,
    category indicators, and unseen-category rate) against the training data at
    geometrically growing checkpoints (16, 32, 64, ...). The j-th checkpoint
    spends alpha / ((j+1)(j+2)) in total, split evenly across all features.
    """
    specs = _build(train_csv)
    if not specs or not rows:
        return None

    F = len(specs)
    window = {key: [] for key, _, _ in specs}
    next_check = _FIRST_CHECK
    j = 0

    for i, row in enumerate(rows):
        for key, _x, get in specs:
            v = get(row)
            if v is not None and np.isfinite(v):
                window[key].append(float(v))

        if i + 1 == next_check:
            alpha_j = _ALPHA / ((j + 1) * (j + 2)) / F
            for key, x, _get in specs:
                y = np.asarray(window[key], dtype=float)
                if len(y) == 0:
                    continue
                if _z_pvalue(x, y) <= alpha_j:
                    return i
            j += 1
            next_check *= 2

    return None

