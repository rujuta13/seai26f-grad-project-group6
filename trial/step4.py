"""step4.py: streaming keep/discard decisions and drift warnings for nightly benchmarks.

stream() keeps a row when no earlier row (training or streamed) is at least as good
on every goal and strictly better on one, i.e. the row is on the Pareto front of
everything seen so far.

drift() warns at the first row where the last _WINDOW new rows sit significantly
further from the training inputs than training rows sit from each other. The
distance is the nearest-neighbour distance in standardised input space, and the
significance test is step3.same().
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _goal_matrix, _load, _to_value
from step3 import same

_WINDOW = 30      # number of recent streamed rows tested together for drift
_CHUNK_D = 256    # chunk size for the pairwise distance computation


def _goal_vector(row, header, maximise, minimise):
    """Goal values of one row as a minimisation vector, in the same order as step1."""
    vals = [float(row[header[i]]) for i in minimise]
    vals += [-float(row[header[i]]) for i in maximise]
    return np.array(vals, dtype=float)


def _encode_train(rows, inputs):
    """Numeric encoding of training inputs plus the mappings needed to encode new rows."""
    n = len(rows)
    X = np.zeros((n, len(inputs)), dtype=float)
    specs = []
    for j, col in enumerate(inputs):
        vals = [_to_value(r[col]) for r in rows]
        if all(isinstance(v, float) for v in vals):
            X[:, j] = vals
            specs.append(("num", float(np.median(vals))))
        else:
            codes = {v: k for k, v in enumerate(sorted({str(v) for v in vals}))}
            X[:, j] = [codes[str(v)] for v in vals]
            specs.append(("cat", codes))
    return X, specs


def _encode_row(row, header, inputs, specs):
    """Encode one new row with the mappings learned from the training data."""
    q = np.zeros(len(inputs), dtype=float)
    for j, col in enumerate(inputs):
        raw = row.get(header[col])
        kind, info = specs[j]
        if kind == "num":
            v = _to_value(raw) if raw is not None else None
            q[j] = v if isinstance(v, float) else info
        else:
            q[j] = info.get(str(raw), -1) if raw is not None else -1
    return q


def _loo_nearest(Z):
    """Distance from each training row to its nearest other training row."""
    n = Z.shape[0]
    out = np.empty(n, dtype=float)
    for s in range(0, n, _CHUNK_D):
        e = min(s + _CHUNK_D, n)
        diff = Z[s:e, None, :] - Z[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        out[s:e] = d.min(axis=1)
    return out.tolist()


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or discard (False) each row, in arrival order.

    A row is kept if no training row and no earlier streamed row is at least as
    good on every goal and strictly better on at least one.
    """
    header, train_rows = _load(train_csv)
    maximise, minimise, _ = _classify(header)
    known = _goal_matrix(train_rows, maximise, minimise)
    if known.shape[1] == 0:
        return [False] * len(rows)

    decisions: list[bool] = []
    for row in rows:
        g = _goal_vector(row, header, maximise, minimise)
        if known.shape[0] == 0:
            dominated = False
        else:
            le = (known <= g).all(axis=1)   # known row is no worse on every goal
            lt = (known < g).any(axis=1)    # and strictly better on some goal
            dominated = bool((le & lt).any())
        decisions.append(not dominated)
        known = np.vstack([known, g[None, :]])
    return decisions


def drift(train_csv, rows) -> int | None:
    """Index of the first streamed row at which the new rows stop resembling train_csv.

    Warns when the last _WINDOW new rows are significantly further from the
    training inputs than training rows are from each other. Returns None if
    no such warning is raised.
    """
    header, train_rows = _load(train_csv)
    _, _, inputs = _classify(header)
    n = len(train_rows)
    if n < 2 or not rows:
        return None

    X, specs = _encode_train(train_rows, inputs)
    if inputs:
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
    else:
        scaler = None
        Z = np.zeros((n, 0), dtype=float)
    ref = _loo_nearest(Z)

    new_dist: list[float] = []
    for i, row in enumerate(rows):
        if inputs:
            q = _encode_row(row, header, inputs, specs)
            zq = scaler.transform(q.reshape(1, -1))[0]
        else:
            zq = np.zeros(0, dtype=float)
        d = float(np.sqrt(((Z - zq) ** 2).sum(axis=1)).min())
        new_dist.append(d)
        if len(new_dist) >= _WINDOW and not same(ref, new_dist[-_WINDOW:]):
            return i
    return None

