"""Step 22: client-facing guess for a single customer.

guess(train_csv, row): estimates every goal for a customer whose goal columns
are removed, returning (estimate, low, high) per goal.

For the client's two questions, read these goals from the result:
  * NUMWEBPURCHASES+ and NUMSTOREPURCHASES+: expected online vs in-store purchases.
  * RESPONSE+: whether the customer answers the next campaign (0 or 1). Its
    estimate and interval are clipped to [0, 1].

The result covers every goal in train_csv, as step12 provides. Wide intervals
mean low confidence.
"""

import csv
import math

import step12


def _goal_columns(train_csv) -> list[str]:
    with open(train_csv, newline="") as f:
        cols = list(csv.DictReader(f).fieldnames or [])
    return [c for c in cols if c.endswith(("+", "-"))]


def _finite_or(value, fallback):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return f if math.isfinite(f) else float(fallback)


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a customer `row` (goal columns removed).

    Returns {goal: (estimate, low, high)} where low..high is the range the true
    value should fall in with fairly high confidence.
    """
    goals = _goal_columns(train_csv)
    raw = step12.guess(train_csv, row)

    out: dict[str, tuple[float, float, float]] = {}
    for g in goals:
        est, low, high = raw.get(g, (0.0, -math.inf, math.inf))
        est = _finite_or(est, 0.0)
        low = _finite_or(low, est)
        high = _finite_or(high, est)

        if g == "RESPONSE+":
            est = min(max(est, 0.0), 1.0)
            low = min(max(low, 0.0), 1.0)
            high = min(max(high, 0.0), 1.0)

        low = min(low, est)
        high = max(high, est)
        out[g] = (est, low, high)

    return out

