"""step5.py: blame the input columns for bad rows.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers, and
the goal normalisation). Steps 2 to 4 are not needed.

A row's badness is 1 minus its mean normalised goal score (step1 scaling,
where 1 is best), so 0 is best on all goals and 1 is worst. An input column
is blamed in proportion to how much of the variation in badness its values
explain (correlation ratio, eta squared, adjusted for the number of levels).
Numeric inputs with many distinct values are grouped into quantile bins, and
the adjustment stops a column from scoring well just because it has many
levels. Ties keep header order.
"""
from __future__ import annotations

import bisect
import math
import os

from step1 import _classify, _goal_scores, _read, _to_float

_MAX_LEVELS = 10  # numeric columns with more distinct values are binned
_BINS = 5         # quantile bins used for those columns


def _is_num(v: float | None) -> bool:
    return v is not None and not math.isnan(v)


def _keys(rows: list[list[str]], j: int) -> list[tuple[str, object]]:
    """Group key per row for input column j.

    Numeric columns with many distinct values are split into quantile bins.
    Other cells are grouped by their numeric value or their text.
    """
    nums = [_to_float(r[j]) for r in rows]
    cells = [r[j].strip() for r in rows]
    valid = [v for v in nums if _is_num(v)]
    distinct = len(set(valid))
    if len(valid) == len(rows) and distinct > _MAX_LEVELS:
        ordered = sorted(valid)
        n = len(ordered)
        cuts = sorted({
            ordered[min(n - 1, (b * n) // _BINS)] for b in range(1, _BINS)
        })
        return [("bin", bisect.bisect_right(cuts, v)) for v in valid]
    return [
        ("v", v) if _is_num(v) else ("s", c)
        for v, c in zip(nums, cells)
    ]


def _adjusted_eta_squared(keys: list, badness: list[float]) -> float:
    """Share of badness variance explained by the groups, adjusted for levels.

    Returns 0 when badness does not vary, when there is only one group, or
    when the groups leave no residual degrees of freedom.
    """
    n = len(badness)
    mean = sum(badness) / n
    total = sum((b - mean) * (b - mean) for b in badness)
    if total <= 0.0:
        return 0.0

    groups: dict[object, list[float]] = {}
    for k, b in zip(keys, badness):
        s = groups.setdefault(k, [0.0, 0])
        s[0] += b
        s[1] += 1
    g = len(groups)
    if g <= 1 or n - g <= 0:
        return 0.0

    between = 0.0
    for total_b, cnt in groups.values():
        d = total_b / cnt - mean
        between += cnt * d * d
    eta = between / total
    adjusted = 1.0 - (1.0 - eta) * (n - 1) / (n - g)
    return max(0.0, adjusted)


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    A column ranks higher the more its values separate the bad rows (high
    badness) from the good ones. Columns with no explanatory power keep
    their header order at the end.
    """
    header, rows = _read(os.fspath(train_csv))
    inputs, goals = _classify(header)
    names = [header[j] for j in inputs]
    if not rows or not inputs or not goals:
        return names

    S = _goal_scores(rows, goals)
    bad = [1.0 - sum(s) / len(s) for s in S]

    scored: list[tuple[float, int]] = []
    for pos, j in enumerate(inputs):
        keys = _keys(rows, j)
        scored.append((-_adjusted_eta_squared(keys, bad), pos))
    scored.sort()
    return [names[pos] for _, pos in scored]

