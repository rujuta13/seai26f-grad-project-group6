import numpy as np

from step1 import _read, _columns, _to_float_column


def _same(a, b):
    """True if two cell values denote the same input value."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs changed as little as possible so the build is good.

    A candidate is a training build that is good by the step6 criterion: no
    other training build beats it on every goal at once, and on average it
    beats at least half of the training builds across the goals. Candidates
    are taken from observed builds, so every returned configuration has known
    goal values. Among candidates reachable without changing any column in
    `frozen`, the one that differs from `row` in the fewest inputs is chosen.
    Ties go to the higher average percentile, then to the lower training index.

    Only input columns are returned. If no admissible good candidate exists in
    the training data, the row's inputs are returned unchanged.
    """
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    frozen_set = set(frozen)
    original = {c: row.get(c) for c in inputs}

    if not rows or not goals or not inputs:
        return original

    names = [g for g, _ in goals]
    G = len(names)
    S = np.column_stack(
        [sign * _to_float_column(rows, name) for name, sign in goals]
    )
    fin = np.all(np.isfinite(S), axis=1)
    idx_fin = np.flatnonzero(fin)
    if idx_fin.size == 0:
        return original
    Sf = S[idx_fin]

    # Pareto status and average percentile (oriented: larger is better).
    nondom = np.ones(len(idx_fin), dtype=bool)
    pct = np.zeros(len(idx_fin), dtype=float)
    for k in range(len(idx_fin)):
        v = Sf[k]
        at_least = np.all(Sf >= v, axis=1)
        strictly = np.any(Sf > v, axis=1)
        nondom[k] = not bool(np.any(at_least & strictly))
        pct[k] = float(np.mean([np.mean(Sf[:, g] < v[g]) for g in range(G)]))

    good = nondom & (pct >= 0.5)
    if not np.any(good):
        return original

    best_key = None
    best_row = None
    for k in np.flatnonzero(good):
        t = rows[int(idx_fin[k])]
        n_changed = 0
        admissible = True
        for c in inputs:
            if not _same(original[c], t[c]):
                if c in frozen_set:
                    admissible = False
                    break
                n_changed += 1
        if not admissible:
            continue
        key = (n_changed, -float(pct[k]), int(idx_fin[k]))
        if best_key is None or key < best_key:
            best_key = key
            best_row = t

    if best_row is None:
        return original

    result = {}
    for c in inputs:
        if _same(original[c], best_row[c]):
            result[c] = original[c]
        else:
            result[c] = best_row[c]
    return result

