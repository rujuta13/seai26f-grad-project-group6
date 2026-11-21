import numpy as np

from step1 import _read, _columns, _to_float_column

_MIN_GROUP = 5


def _badness(rows, goals):
    """Per-row badness: higher means worse on the goals (z-scored, oriented)."""
    n = len(rows)
    cols = []
    for name, sign in goals:
        y = sign * _to_float_column(rows, name)
        fin = np.isfinite(y)
        if not np.any(fin):
            cols.append(np.zeros(n))
            continue
        mu = float(y[fin].mean())
        sd = float(y[fin].std())
        if sd <= 0.0:
            sd = 1.0
        z = np.zeros(n)
        z[fin] = (y[fin] - mu) / sd
        cols.append(z)
    Z = np.column_stack(cols)
    # Oriented goals are larger-is-better, so badness is the negated mean.
    return -Z.mean(axis=1)


def _score(values, bad):
    """Blame score for one input column: how much worse rows get when it takes a value."""
    n = len(bad)
    uniq = np.unique(values)
    if len(uniq) <= 1:
        return 0.0

    if set(uniq.tolist()) <= {0.0, 1.0}:
        # Binary pass flag: blame it when enabling it makes builds worse.
        on = values == 1.0
        off = ~on
        if on.sum() < _MIN_GROUP or off.sum() < _MIN_GROUP:
            return 0.0
        return float(bad[on].mean() - bad[off].mean())

    # Categorical or multi-valued: take the value whose rows are worst
    # relative to the rest, among values with enough support.
    best = None
    for v in uniq:
        m = values == v
        k = int(m.sum())
        if k < _MIN_GROUP or n - k < _MIN_GROUP:
            continue
        d = float(bad[m].mean() - bad[~m].mean())
        if best is None or d > best:
            best = d
    return best if best is not None else 0.0


def blame(train_csv) -> list[str]:
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    if not inputs:
        return []
    if not rows or not goals:
        return list(inputs)

    bad = _badness(rows, goals)
    scores = []
    for c in inputs:
        values = _to_float_column(rows, c)
        scores.append(_score(values, bad))

    order = sorted(range(len(inputs)), key=lambda j: (-scores[j], j))
    return [inputs[j] for j in order]

