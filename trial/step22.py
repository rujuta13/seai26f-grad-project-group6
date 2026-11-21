"""Step 22: guess a new customer's goal values from his other numbers, with a range.

guess(train_csv, row) estimates each goal for a new customer from the training
customers whose input columns are closest to his. The goals include online
purchases (NUMWEBPURCHASES+), store purchases (NUMSTOREPURCHASES+) and campaign
response (RESPONSE+), along with the other goals in the file.

For each goal, the estimate is the median of the nearest customers' values,
and low..high is their 10th to 90th percentile range. A narrow range means the
neighbours agree, so the guess is confident. A wide range means they disagree,
so the guess is shaky.

Numeric inputs are compared on their training range, so one large-scale column
does not dominate the match. Missing cells are penalised but not fatal. If the
row shares no usable inputs with the training data, every training customer is
used.
"""
import math

from step1 import _load

_MIN_VALUES = 3          # fewer neighbour values than this: fall back to all training values
_MISSING_PENALTY = 0.5   # distance added when exactly one of two cells is missing


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _cell(v):
    """Comparable input cell: finite float, non-blank string, or None."""
    f = _finite(v)
    if f is not None:
        return f
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None


def _finite_values(cells):
    """Sorted list of the finite float values among cells."""
    return sorted(f for f in (_finite(v) for v in cells) if f is not None)


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _spans(rows, cols):
    """Training range of each numeric input column (0.0 if none)."""
    out = {}
    for c in cols:
        nums = _finite_values(r.get(c) for r in rows)
        out[c] = (nums[-1] - nums[0]) if nums else 0.0
    return out


def _distance(query, other, inputs, spans):
    """Distance between the query cells and a training row over the shared inputs."""
    d = 0.0
    for c in inputs:
        a = query[c]
        b = _cell(other.get(c))
        if a is None and b is None:
            continue
        if a is None or b is None:
            d += _MISSING_PENALTY
        elif isinstance(a, float) and isinstance(b, float):
            s = spans[c]
            if s > 0.0:
                d += min(1.0, abs(a - b) / s)
            elif a != b:
                d += 1.0
        elif a != b:
            d += 1.0
    return d


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal in the training file.

    `row` is a dict of input columns only (goal columns removed). The estimate
    is the median goal value among the nearest training customers, and low..high
    is their 10th..90th percentile range. Goals with no usable training values
    get (0.0, 0.0, 0.0).
    """
    header, rows = _load(train_csv)
    goals = [c for c in header if _is_goal(c)]
    if not goals:
        return {}

    n = len(rows)
    if n == 0:
        return {g: (0.0, 0.0, 0.0) for g in goals}

    inputs = [c for c in header if _is_input(c) and c in row]
    spans = _spans(rows, inputs)
    query = {c: _cell(row.get(c)) for c in inputs}

    if any(v is not None for v in query.values()):
        dists = [_distance(query, r, inputs, spans) for r in rows]
        order = sorted(range(n), key=lambda i: (dists[i], i))
        k = min(n, max(10, int(math.sqrt(n))))
        near = order[:k]
    else:
        # Nothing to match on: every training customer is a candidate.
        near = list(range(n))

    out = {}
    for g in goals:
        vals = _finite_values(rows[i].get(g) for i in near)
        if len(vals) < _MIN_VALUES:
            vals = _finite_values(r.get(g) for r in rows)
        if not vals:
            out[g] = (0.0, 0.0, 0.0)
            continue

        est = _quantile(vals, 0.5)
        low = min(_quantile(vals, 0.1), est)
        high = max(_quantile(vals, 0.9), est)
        out[g] = (float(est), float(low), float(high))

    return out

