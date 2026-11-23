import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _column_kind, _load
from step5 import _badness
from step15 import _encode_blocks


_N_TREES = 200
_MIN_LEAF = 2
_SEED = 0
_REPEATS = 5
# The client's two questions concern these goals: response to campaigns and complaints.
_FOCUS_GOALS = ("RESPONSE+", "COMPLAIN-")


def _goal_columns(header):
    goals = [c for c in header if _column_kind(c) in ("max", "min")]
    focus = [c for c in goals if c in _FOCUS_GOALS]
    return focus if focus else goals


def _signed_goals(cols, rows):
    """(n, m) matrix where smaller is always better; missing or non-finite goals are worst."""
    G = np.empty((len(rows), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        sign = -1.0 if _column_kind(c) == "max" else 1.0
        for i, r in enumerate(rows):
            try:
                f = float(r.get(c))
            except (TypeError, ValueError):
                f = math.nan
            G[i, j] = sign * f if math.isfinite(f) else math.inf
    return G


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Bad rows are judged on the goals RESPONSE+ (never responding to campaigns)
    and COMPLAIN- (complaining); if neither is present, on all goal columns.
    Each row gets a badness score (mean normalised rank over those goals, where
    larger is worse). A random forest is fitted from the input columns to that
    score. Each input's importance is the average increase in squared error
    when its values are shuffled across rows (permutation importance).
    Identifier-like inputs are ranked last. Ties are broken by header order.
    """
    header, rows = _load(train_csv)
    inputs = [c for c in header if _column_kind(c) == "input"]
    n = len(rows)

    if n == 0 or not inputs:
        return list(inputs)

    goal_cols = _goal_columns(header)
    if not goal_cols:
        return list(inputs)

    score = _badness(_signed_goals(goal_cols, rows))
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
    usable = sorted(importance, key=lambda c: (-importance[c], position[c]))
    return usable + [c for c in inputs if c in dropped]

