"""Step 4: flow control for nightly benchmark results.

stream(): keeps a new row when its normalised goal total reaches the 95th
percentile of the training rows' totals. Normalisation follows step1.best:
each goal is scaled to [0, 1] over the training range, minimised goals are
negated, and missing values count as worst.

drift(): compares a sliding window of recent goal values with the training
goal values using step3.same. It checks every _STRIDE rows and warns once
_CONFIRM consecutive checks report a real difference, which limits false alarms.
"""

import math
from functools import lru_cache

import step3
from step1 import _read, _classify

_KEEP_QUANTILE = 0.95  # keep rows scoring at or above this training quantile
_WINDOW = 40           # recent rows compared against training in drift()
_STRIDE = 10           # run a drift check every _STRIDE rows
_CONFIRM = 3           # consecutive significant checks before warning


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _score(row, ranges):
    """Sum of normalised goal values (higher is better). Missing counts as 0."""
    total = 0.0
    for name, sign, lo, span in ranges:
        v = row.get(name)
        if not _is_num(v):
            continue  # missing counts as worst
        x = v * sign
        total += (x - lo) / span if span > 0 else 1.0
    return total


@lru_cache(maxsize=4)
def _model(train_csv):
    """Parse train_csv once.

    Returns (goal_names, ranges, train_rows, threshold):
      goal_names: goal columns as they appear in the header.
      ranges: (name, sign, lo, span) per goal, over the training data.
      train_rows: the training rows.
      threshold: keep cut-off on the normalised totals, or None if no goals/rows.
    """
    header, rows = _read(train_csv)
    _, goals_max, goals_min = _classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]

    ranges = []
    for name, sign in goals:
        vals = [r[name] * sign for r in rows if _is_num(r.get(name))]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 0.0
        ranges.append((name, sign, lo, hi - lo))
    ranges = tuple(ranges)
    names = tuple(g for g, _ in goals)

    if not goals or not rows:
        threshold = None
    else:
        totals = sorted(_score(r, ranges) for r in rows)
        threshold = _quantile(totals, _KEEP_QUANTILE)
    return names, ranges, rows, threshold


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or drop (False) each incoming row, in order.

    Each decision uses only the training data and the row itself: the row is
    kept when its normalised goal total reaches the training threshold.
    """
    _, ranges, _, threshold = _model(train_csv)
    if threshold is None:
        return [False] * len(rows)
    return [_score(r, ranges) >= threshold for r in rows]


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the new results have stopped
    looking like the training results, or None if they never do.

    At each periodic check, the most recent _WINDOW rows are tested per goal
    against the training goal values with step3.same. A warning is raised once
    _CONFIRM consecutive checks find a real difference.
    """
    names, _, train_rows, _ = _model(train_csv)
    if not names or not train_rows:
        return None
    ref = {g: [r[g] for r in train_rows if _is_num(r.get(g))] for g in names}

    streak = 0
    for i in range(_WINDOW - 1, len(rows)):
        if (i + 1) % _STRIDE:
            continue
        window = rows[i + 1 - _WINDOW: i + 1]
        differs = any(
            not step3.same([r.get(g) for r in window], ref[g])
            for g in names
        )
        streak = streak + 1 if differs else 0
        if streak >= _CONFIRM:
            return i
    return None

