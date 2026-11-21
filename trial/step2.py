import numpy as np
from sklearn.neighbors import NearestNeighbors

from step1 import _read, _columns, _to_float_column


def _encode_inputs(rows, inputs):
    """Build per-column encoders from training rows, returning (matrix, encode_row)."""
    specs = []  # per input column: ("num", None) or ("cat", mapping dict)
    cols = []
    for c in inputs:
        values = [r[c] for r in rows]
        try:
            col = np.array([float(v) for v in values], dtype=float)
            specs.append(("num", None))
        except (TypeError, ValueError):
            mapping = {}
            for v in values:
                if v not in mapping:
                    mapping[v] = float(len(mapping))
            col = np.array([mapping[v] for v in values], dtype=float)
            specs.append(("cat", mapping))
        cols.append(col)

    X = np.column_stack(cols) if cols else np.zeros((len(rows), 0))
    col_means = X.mean(axis=0) if X.shape[1] else np.zeros(0)

    def encode_row(row_vals):
        out = np.empty(len(inputs), dtype=float)
        for j, (c, (kind, mapping)) in enumerate(zip(inputs, specs)):
            v = row_vals.get(c)
            if kind == "num":
                try:
                    out[j] = float(v)
                except (TypeError, ValueError):
                    out[j] = col_means[j]
                if not np.isfinite(out[j]):
                    out[j] = col_means[j]
            else:
                if v in mapping:
                    out[j] = mapping[v]
                else:
                    # Unseen category: fall back to the training mean code.
                    out[j] = col_means[j]
        return out

    return X, encode_row


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    n = len(rows)

    if n == 0:
        return {name: (float("nan"), float("-inf"), float("inf"))
                for name, _ in goals}

    X, encode_row = _encode_inputs(rows, inputs)

    # Standardise inputs so every feature contributes comparably to distance.
    if X.shape[1]:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (X - mu) / sd
        z = (encode_row(row) - mu) / sd
    else:
        Z = np.zeros((n, 1))
        z = np.zeros(1)

    k = min(15, n)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, idx = nn.kneighbors(z.reshape(1, -1))
    idx = idx[0]

    result: dict[str, tuple[float, float, float]] = {}
    for name, _sign in goals:
        Y = _to_float_column(rows, name)
        neighbours = Y[idx]
        estimate = float(np.median(neighbours))
        low = float(np.percentile(neighbours, 10))
        high = float(np.percentile(neighbours, 90))
        result[name] = (estimate, low, high)

    return result

