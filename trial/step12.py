import numpy as np
from sklearn.neighbors import NearestNeighbors

from step1 import _read, _columns
from step2 import _encode_inputs

_K = 25
_MIN_LOCAL = 3


def _goal_values(rows, name):
    """Finite float values of a goal column, NaN where unusable."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        try:
            v = float(r.get(name))
        except (TypeError, ValueError, AttributeError):
            continue
        if np.isfinite(v):
            out[i] = v
    return out


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate each goal for `row` from similar training builds.

    `row` has no goal columns; any goal keys present are ignored. Neighbours are
    the training rows closest in standardised input space. For each goal the
    estimate is the median of the neighbours' finite goal values, and low..high
    is their 10th to 90th percentile range. If too few neighbours have a finite
    value for a goal, the whole training column is used instead.
    """
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    n = len(rows)

    if n == 0 or not goals:
        return {name: (float("nan"), float("-inf"), float("inf"))
                for name, _ in goals}

    # Input features only; goal columns of `row` are never read.
    if inputs:
        X, encode_row = _encode_inputs(rows, inputs)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (X - mu) / sd
        z = (encode_row(row) - mu) / sd
    else:
        Z = np.zeros((n, 1))
        z = np.zeros(1)

    k = min(_K, n)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, idx = nn.kneighbors(z.reshape(1, -1))
    idx = idx[0]

    result: dict[str, tuple[float, float, float]] = {}
    for name, _sign in goals:
        Y = _goal_values(rows, name)
        local = Y[idx]
        local = local[np.isfinite(local)]
        if local.size < _MIN_LOCAL:
            local = Y[np.isfinite(Y)]
        if local.size == 0:
            result[name] = (float("nan"), float("-inf"), float("inf"))
            continue
        estimate = float(np.median(local))
        low = float(np.percentile(local, 10))
        high = float(np.percentile(local, 90))
        result[name] = (estimate, low, high)

    return result

