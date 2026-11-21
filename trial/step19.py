"""Step 19: spot rows that look like data-entry mistakes or are unlike the training data.

odd(train_csv, rows) returns one suspicion score per row, goals included.
Higher means more likely a glitch, a typo, or a build unlike anything in
train_csv. Two parts are added:

1. Goal glitches. Each goal is centred on its training median and scaled by
   its training median absolute deviation (times 1.4826, so it matches a
   standard deviation for normal data). The part of a robust z-score beyond
   _TOL is added to the score. A goal cell that is missing, non-numeric or
   not finite adds _BAD_CELL, since a measurement that did not parse is a
   likely glitch.
2. Input novelty. The row's inputs are compared with every training row
   (numeric inputs are scaled by their training range, other values must
   match exactly). The distance to the nearest training row is divided by
   the typical leave-one-out nearest-neighbour distance among training rows,
   so about 1 is ordinary and larger means unlike anything measured.

The scores rank rows for checking first. They are not calibrated probabilities.
"""
import math

from step1 import _load

_TOL = 3.0          # robust z-scores up to this are normal variation
_BAD_CELL = 10.0    # penalty for a goal cell that is missing or not a finite number
_PROBES = 200       # training rows used to estimate typical nearest-neighbour distance
_MAD_TO_SD = 1.4826 # scales a median absolute deviation to a standard deviation
_MEAN_AD_TO_SD = 1.2533


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not (_is_goal(c) or c.endswith("X"))


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _level(v):
    """Comparable key for an input cell: finite float, non-blank string, or None."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _median(vals):
    """Median of a non-empty list."""
    s = sorted(vals)
    m = len(s)
    mid = m // 2
    if m % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _goal_stats(train, goals):
    """Return {goal: (median, robust scale)} from the training rows."""
    stats = {}
    for c in goals:
        vals = [f for f in (_finite(r.get(c)) for r in train) if f is not None]
        if not vals:
            continue
        med = _median(vals)
        scale = _MAD_TO_SD * _median([abs(v - med) for v in vals])
        if scale <= 0.0:
            # Fall back to mean absolute deviation, scaled to sd for normal data.
            scale = _MEAN_AD_TO_SD * sum(abs(v - med) for v in vals) / len(vals)
        if scale <= 0.0:
            # Constant column: any real change is still a large deviation.
            scale = 0.01 * max(abs(med), 1.0)
        stats[c] = (med, scale)
    return stats


def _goal_part(r, goals, stats):
    """Goal-glitch part of the suspicion score for one row."""
    total = 0.0
    for c in goals:
        f = _finite(r.get(c))
        if f is None:
            total += _BAD_CELL
            continue
        if c not in stats:
            continue
        med, scale = stats[c]
        z = abs(f - med) / scale
        if z > _TOL:
            total += z - _TOL
    return total


def _spans(train, inputs):
    """Training range of each numeric input column (0.0 if none)."""
    spans = {}
    for c in inputs:
        nums = [v for v in (_level(r.get(c)) for r in train) if isinstance(v, float)]
        spans[c] = (max(nums) - min(nums)) if nums else 0.0
    return spans


def _dist(a, b, inputs, spans):
    """Mean per-column difference in [0, 1] between two level dicts."""
    if not inputs:
        return 0.0
    d = 0.0
    for c in inputs:
        x = a[c]
        y = b[c]
        if isinstance(x, float) and isinstance(y, float):
            s = spans[c]
            if s > 0.0:
                d += min(1.0, abs(x - y) / s)
            elif x != y:
                d += 1.0
        elif x != y:
            d += 1.0
    return d / len(inputs)


def _typical_nn(tl, inputs, spans):
    """Typical leave-one-out nearest-neighbour distance among training rows.

    Estimated on a spread of probe rows and floored so duplicate flag sets
    do not make the novelty ratio blow up.
    """
    n = len(tl)
    if not inputs or n == 0:
        return 1.0
    step = max(1, n // _PROBES)
    probes = list(range(0, n, step))[:_PROBES]
    nn = []
    for p in probes:
        best = math.inf
        for j in range(n):
            if j != p:
                d = _dist(tl[p], tl[j], inputs, spans)
                if d < best:
                    best = d
        if best < math.inf:
            nn.append(best)
    floor = 0.5 / len(inputs)
    return max(_median(nn) if nn else 0.0, floor)


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included), in input order.

    Higher means more likely a glitch, a typo, or unlike anything in
    train_csv. Each score is the sum of a goal-glitch part and an input
    novelty part (see the module docstring). An empty `rows` gives [].
    """
    header, train = _load(train_csv)
    rows = list(rows)
    if not rows:
        return []

    goals = [c for c in header if _is_goal(c)]
    inputs = [c for c in header if _is_input(c)]

    stats = _goal_stats(train, goals)
    spans = _spans(train, inputs)
    tl = [{c: _level(r.get(c)) for c in inputs} for r in train]
    typical = _typical_nn(tl, inputs, spans)

    out = []
    for r in rows:
        total = _goal_part(r, goals, stats)
        if inputs and tl:
            rl = {c: _level(r.get(c)) for c in inputs}
            nn = min(_dist(rl, t, inputs, spans) for t in tl)
            total += nn / typical
        out.append(float(total))
    return out

