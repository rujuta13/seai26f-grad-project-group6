"""Step 22: guess online vs in-store purchases and campaign response for a new customer."""

import numpy as np

from step1 import _read, _columns

_K = 25
_MIN_LOCAL = 3


def _parse(v):
    """Finite float for a cell value, NaN otherwise."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _key(v):
    """Category key for a cell value."""
    return "" if v is None else str(v)


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate each goal for a new customer from similar training customers.

    `row` has no goal columns; any goal keys present are ignored. Customers are
    compared on their inputs: numeric inputs are standardised, and categorical
    inputs become indicator columns. The neighbours are the 25 closest training
    rows. For each goal, low..high is the 10th to 90th percentile of the
    neighbours' finite values. For 0/1 goals such as campaign response, the
    estimate is the neighbours' rate of 1s; for count goals it is the median.
    If too few neighbours have a finite value, the whole training column is used.
    """
    fieldnames, train_rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    n = len(train_rows)

    if n == 0 or not goals:
        return {name: (float("nan"), float("-inf"), float("inf"))
                for name, _ in goals}

    # Per-input encoding spec, learned from training rows only.
    specs = []  # (kind, column, payload, mu, sd)
    for c in inputs:
        raw = [r.get(c) for r in train_rows]
        parsed = np.array([_parse(v) for v in raw], dtype=float)
        ok = np.isfinite(parsed)
        if ok.sum() >= max(1, 0.9 * n):
            med = float(np.median(parsed[ok]))
            filled = np.where(ok, parsed, med)
            mu = float(filled.mean())
            sd = float(filled.std())
            if not sd > 0:
                sd = 1.0
            specs.append(("num", c, med, mu, sd))
        else:
            levels = list(dict.fromkeys(_key(v) for v in raw))
            specs.append(("cat", c, levels, None, None))

    def encode(r):
        out = []
        for kind, c, payload, mu, sd in specs:
            v = r.get(c)
            if kind == "num":
                x = _parse(v)
                if not np.isfinite(x):
                    x = payload
                out.append((x - mu) / sd)
            else:
                k = _key(v)
                out.extend(1.0 if k == lev else 0.0 for lev in payload)
        return out

    if specs:
        Z = np.array([encode(r) for r in train_rows], dtype=float)
        z = np.array(encode(row), dtype=float)
    else:
        Z = np.zeros((n, 1))
        z = np.zeros(1)

    dist = np.sqrt(((Z - z) ** 2).sum(axis=1))
    k = min(_K, n)
    idx = np.argsort(dist, kind="stable")[:k]

    result: dict[str, tuple[float, float, float]] = {}
    for name, _sign in goals:
        Y = np.array([_parse(r.get(name)) for r in train_rows], dtype=float)
        Yf = Y[np.isfinite(Y)]
        if Yf.size == 0:
            result[name] = (float("nan"), float("-inf"), float("inf"))
            continue

        local = Y[idx]
        local = local[np.isfinite(local)]
        if local.size < _MIN_LOCAL:
            local = Yf

        is_binary = set(np.unique(Yf).tolist()) <= {0.0, 1.0}
        if is_binary:
            estimate = float(local.mean())
        else:
            estimate = float(np.median(local))
        low = float(np.percentile(local, 10))
        high = float(np.percentile(local, 90))
        result[name] = (estimate, low, high)

    return result

