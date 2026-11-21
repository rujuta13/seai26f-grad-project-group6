"""step5.py: blame. Which input columns are most associated with bad rows."""

import numpy as np

from step1 import _load
from step3 import same

# Rows at or below this quantile of the combined goal score count as "bad".
_BAD_QUANTILE = 0.25


def _goal_scores(rows, goals):
    """Combined goal score per row, larger is better (z-scored, minimised goals negated)."""
    M = np.array(
        [[float(r[name]) * sign for name, sign in goals] for r in rows],
        dtype=float,
    )
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    sd = M.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (M - M.mean(axis=0)) / sd
    return Z.sum(axis=1)


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    A column is blamed in proportion to how differently its values are
    distributed among the bad rows compared with the good rows (total variation
    distance). Columns whose difference is statistically real (step3.same) rank
    ahead of those that look like noise.
    """
    rows, inputs, goals = _load(train_csv)
    n = len(rows)
    if not inputs:
        return []
    if not goals or n < 2:
        return list(inputs)

    score = _goal_scores(rows, goals)
    cut = np.quantile(score, _BAD_QUANTILE)
    bad = [i for i in range(n) if score[i] <= cut]
    good = [i for i in range(n) if score[i] > cut]
    if not bad or not good:
        # Goal scores are too tied to separate bad from good rows.
        return list(inputs)

    tv_scores: dict[str, float] = {}
    significant: dict[str, bool] = {}
    for c in inputs:
        vals = [str(r[c]) for r in rows]
        tv = 0.0
        sig = False
        for v in sorted(set(vals)):
            ind = np.array([1.0 if x == v else 0.0 for x in vals], dtype=float)
            p_bad = float(ind[bad].mean())
            p_good = float(ind[good].mean())
            tv += abs(p_bad - p_good)
            if not same(ind[bad], ind[good]):
                sig = True
        tv_scores[c] = 0.5 * tv
        significant[c] = sig

    order = {c: i for i, c in enumerate(inputs)}
    return sorted(
        inputs,
        key=lambda c: (not significant[c], -tv_scores[c], order[c]),
    )

