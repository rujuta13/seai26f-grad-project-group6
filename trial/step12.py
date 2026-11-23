"""Step 12: client-facing guesser for a single build.

guess(train_csv, row): estimates every goal for a player whose goal columns
are removed, returning (estimate, low, high) per goal. The low..high range is
the interval the true value should fall in with fairly high confidence. A
wide range means low confidence.
"""

import csv
import math

import numpy as np

import step2


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a flag set `row` (goal columns removed).

    Returns {goal: (estimate, low, high)}, where low..high is a range the true
    value should fall in with fairly high confidence. Built on step2.guess,
    with non-finite or inverted values repaired from the training data.
    """
    cols, rows = _load(train_csv)
    goals = [c for c in cols if c.endswith(("+", "-"))]

    raw = step2.guess(train_csv, row)

    out: dict[str, tuple[float, float, float]] = {}
    for g in goals:
        train_vals = np.array(
            [_to_float(r.get(g)) for r in rows], dtype=float
        )
        train_vals = train_vals[np.isfinite(train_vals)]

        if train_vals.size:
            med = float(np.median(train_vals))
            q_lo, q_hi = (float(q) for q in np.quantile(train_vals, [0.05, 0.95]))
        else:
            med, q_lo, q_hi = 0.0, 0.0, 0.0

        est, low, high = raw.get(g, (med, -math.inf, math.inf))

        if not math.isfinite(est):
            est = med
        if not math.isfinite(low):
            low = min(q_lo, est)
        if not math.isfinite(high):
            high = max(q_hi, est)

        low = min(float(low), float(est))
        high = max(float(high), float(est))
        out[g] = (float(est), low, high)

    return out

