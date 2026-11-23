import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _column_kind, _goal_matrix, _load
from step5 import _badness


_N_TREES = 200
_MIN_LEAF = 2
_SEED = 0
_REPEATS = 5          # permutations per input when measuring importance
_ID_FRACTION = 0.9    # a column this unique is treated as an identifier
_MAX_LEVELS = 50      # categorical columns with more levels are skipped


def _num(cell):
    """Return cell as a finite float, or None if missing or not numeric."""
    if cell is None or isinstance(cell, bool):
        return None
    try:
        f = float(cell)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _encode_blocks(inputs, rows):
    """Turn each usable input column into a block of model columns.

    Numeric columns become one standardisation-free column (missing cells
    filled with the column mean). Categorical columns are one-hot encoded so
    no spurious ordering is imposed. Identifier-like columns (names, urls,
    row numbers) and very high-cardinality columns are dropped.
    Returns (blocks, dropped) where blocks is a list of (name, matrix).
    """
    n = len(rows)
    blocks = []
    dropped = []
    for c in inputs:
        raw = [r.get(c, "") for r in rows]
        nums = [_num(v) for v in raw]
        present = [v for v in nums if v is not None]
        numeric = bool(present) and len(present) >= 0.9 * n
        distinct = len({str(v) for v in raw})

        if distinct > _ID_FRACTION * n and (
            not numeric or all(float(v).is_integer() for v in present)
        ):
            dropped.append(c)
            continue

        if numeric:
            fill = float(np.mean(present))
            col = np.asarray(
                [fill if v is None else v for v in nums], dtype=float
            )
            blocks.append((c, col[:, None]))
            continue

        levels = sorted({str(v) for v in raw})
        if len(levels) > _MAX_LEVELS:
            dropped.append(c)
            continue
        index = {lab: j for j, lab in enumerate(levels)}
        M = np.zeros((n, len(levels)), dtype=float)
        for i, v in enumerate(raw):
            M[i, index[str(v)]] = 1.0
        blocks.append((c, M))
    return blocks, dropped


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Each row gets a badness score (mean normalised rank over all goals, where
    larger is worse). A random forest is fitted from the input columns to that
    score. Each input's importance is the average increase in squared error
    when its values are shuffled across rows (permutation importance), which
    is less biased toward high-cardinality columns than impurity importance.
    Identifier-like inputs are ranked last. Ties are broken by header order.
    """
    header, rows = _load(train_csv)
    inputs = [c for c in header if _column_kind(c) == "input"]
    n = len(rows)

    if n == 0 or not inputs:
        return list(inputs)

    G = _goal_matrix(header, rows)
    if G.shape[1] == 0:
        return list(inputs)

    score = _badness(G)
    if n < 2 or float(score.std()) == 0.0:
        return list(inputs)

    blocks, dropped = _encode_blocks(inputs, rows)
    if not blocks:
        return list(inputs)

    X = np.hstack([M for _, M in blocks])
    model = RandomForestRegressor(
        n_estimators=_N_TREES,
        min_samples_leaf=_MIN_LEAF,
        random_state=_SEED,
        n_jobs=1,
    ).fit(X, score)

    base_err = float(np.mean((model.predict(X) - score) ** 2))

    rng = np.random.default_rng(_SEED)
    importance = {}
    offset = 0
    for name, M in blocks:
        width = M.shape[1]
        s, e = offset, offset + width
        offset = e
        total = 0.0
        for _ in range(_REPEATS):
            perm = rng.permutation(n)
            Xp = X.copy()
            Xp[:, s:e] = X[perm, s:e]
            err = float(np.mean((model.predict(Xp) - score) ** 2))
            total += err - base_err
        importance[name] = total / _REPEATS

    position = {c: j for j, c in enumerate(inputs)}
    usable = sorted(
        importance,
        key=lambda c: (-importance[c], position[c]),
    )
    return usable + [c for c in inputs if c in dropped]

