import numpy as np

from step1 import _read, _columns, _to_float_column

_ALPHA = 0.05
_N_PERM = 2000
_SEED = 0
_CHUNK = 500
_FIRST_CHECK = 16


def _train_goal_matrix(train_csv):
    """Return (goal names, signs, oriented finite training matrix)."""
    fieldnames, rows = _read(train_csv)
    _, goals = _columns(fieldnames)
    names = [name for name, _ in goals]
    signs = np.array([sign for _, sign in goals], dtype=float)
    if not goals or not rows:
        return names, signs, np.empty((0, len(goals)))
    S = np.column_stack(
        [sign * _to_float_column(rows, name) for name, sign in goals]
    )
    S = S[np.all(np.isfinite(S), axis=1)]
    return names, signs, S


def _oriented_row(row, names, signs):
    """Oriented goal vector for one incoming row, or None if unusable."""
    vals = np.empty(len(names), dtype=float)
    for j, name in enumerate(names):
        try:
            vals[j] = float(row.get(name))
        except (TypeError, ValueError, AttributeError):
            return None
    if not np.all(np.isfinite(vals)):
        return None
    return signs * vals


def stream(train_csv, rows) -> list[bool]:
    """Keep a streamed row iff no reference row dominates it.

    The reference set is the finite training rows plus every row kept so far.
    Dominated rows never change the Pareto status of other rows, so only kept
    rows need to be added to the reference set.
    """
    names, signs, ref = _train_goal_matrix(train_csv)
    decisions: list[bool] = []
    if not names:
        return [False] * len(rows)

    for row in rows:
        v = _oriented_row(row, names, signs)
        if v is None:
            decisions.append(False)
            continue
        if ref.shape[0] == 0:
            keep = True
        else:
            at_least = np.all(ref >= v, axis=1)
            strictly = np.any(ref > v, axis=1)
            keep = not bool(np.any(at_least & strictly))
        decisions.append(keep)
        if keep:
            ref = np.vstack([ref, v[None, :]])
    return decisions


def _perm_pvalue(x, y):
    """Two-sided permutation p-value for the difference in means."""
    n, m = len(x), len(y)
    observed = abs(float(x.mean()) - float(y.mean()))
    if observed == 0.0:
        return 1.0
    pooled = np.concatenate([x, y])
    total = n + m
    rng = np.random.default_rng(_SEED)
    threshold = observed * (1.0 - 1e-12)
    count = 0
    done = 0
    while done < _N_PERM:
        b = min(_CHUNK, _N_PERM - done)
        perm = rng.random((b, total)).argsort(axis=1)
        vals = pooled[perm]
        diff = np.abs(vals[:, :n].mean(axis=1) - vals[:, n:].mean(axis=1))
        count += int(np.count_nonzero(diff >= threshold))
        done += b
    return (count + 1) / (_N_PERM + 1)


def drift(train_csv, rows) -> int | None:
    """Warn at the first row where new results differ from training results.

    Checks happen at geometrically growing window sizes (16, 32, 64, ...), so
    the number of tests is logarithmic in the stream length. The j-th check
    (0-based) spends alpha * 1 / ((j+1)(j+2)) in total, which sums to alpha
    over all checks. The budget is split evenly across goals. Each check
    compares the window of new goal values against the training goal values
    with a permutation test on the difference in means.
    """
    names, _signs, _ref = _train_goal_matrix(train_csv)
    if not names:
        return None

    fieldnames, train_rows = _read(train_csv)
    train_vals = {}
    for name in names:
        col = _to_float_column(train_rows, name)
        train_vals[name] = col[np.isfinite(col)]

    window = {name: [] for name in names}
    G = len(names)
    next_check = _FIRST_CHECK
    j = 0
    count = 0

    for i, row in enumerate(rows):
        count += 1
        for name in names:
            try:
                v = float(row.get(name))
            except (TypeError, ValueError, AttributeError):
                continue
            if np.isfinite(v):
                window[name].append(v)

        if count == next_check:
            alpha_j = _ALPHA / ((j + 1) * (j + 2)) / G
            for name in names:
                x = train_vals[name]
                y = np.asarray(window[name], dtype=float)
                if len(x) == 0 or len(y) == 0:
                    continue
                if _perm_pvalue(x, y) <= alpha_j:
                    return i
            j += 1
            next_check *= 2

    return None

