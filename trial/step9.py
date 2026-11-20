"""Step 9: flag benchmark rows that look like glitches or are unlike anything seen.

odd(train_csv, rows) gives one suspicion score per row, goals included. Higher
means more likely a glitch, a typo, or a flag set unlike anything in the
training data. Two parts are added:

1. Goal glitches. Each goal is centred on its training median and scaled by
   its training median absolute deviation (MAD, times 1.4826 so it matches a
   standard deviation for normal data). A goal value further than _TOL scales
   from the median adds the excess. A goal cell that is missing, non-numeric
   or not finite adds _BAD_CELL, since a measurement that did not parse is a
   likely glitch.
2. Input novelty. The row's flags are compared with every training flag set
   (numbers are scaled by their training range, other values must match
   exactly). The distance to the nearest training flag set is divided by the
   typical nearest-neighbour distance among training rows, so about 1 is
   ordinary and larger means the flag set is unlike anything measured.

The scores are heuristics for ranking which rows to check first, not
calibrated probabilities.
"""
import math

from step1 import _load

_TOL = 3.0         # robust z-scores up to this are treated as normal variation
_BAD_CELL = 10.0   # penalty for a goal cell that is missing or not a finite number
_PROBES = 200      # training rows used to estimate typical nearest-neighbour distance


def _is_input(c):
    return not (c.endswith("+") or c.endswith("-") or c.endswith("X"))


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _level(v):
    """Comparable key for a cell: finite float, non-empty string, or None."""
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

    goals = [c for c in header if c.endswith("+") or c.endswith("-")]
    inputs = [c for c in header if _is_input(c)]

    # Goal centre and robust scale from the training rows.
    gstats = {}
    for c in goals:
        vals = []
        for r in train:
            f = _finite(r.get(c))
            if f is not None:
                vals.append(f)
        if not vals:
            continue
        med = _median(vals)
        scale = 1.4826 * _median([abs(v - med) for v in vals])
        if scale <= 0.0:
            # Fall back to mean absolute deviation (scaled to sd for normal data).
            scale = 1.2533 * sum(abs(v - med) for v in vals) / len(vals)
        if scale <= 0.0:
            # Constant column: any real change is still a large deviation.
            scale = 0.01 * max(abs(med), 1.0)
        gstats[c] = (med, scale)

    def goal_part(r):
        total = 0.0
        for c in goals:
            if c not in gstats:
                continue
            f = _finite(r.get(c))
            if f is None:
                total += _BAD_CELL
                continue
            med, scale = gstats[c]
            z = abs(f - med) / scale
            if z > _TOL:
                total += z - _TOL
        return total

    # Ranges of numeric input columns, for scaling distances.
    spans = {}
    for c in inputs:
        nums = []
        for r in train:
            v = _level(r.get(c))
            if isinstance(v, float):
                nums.append(v)
        spans[c] = (max(nums) - min(nums)) if nums else 0.0

    def dist(a, b):
        """Mean per-column difference in [0, 1] between two level dicts."""
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

    tl = [{c: _level(r.get(c)) for c in inputs} for r in train]
    n = len(tl)

    # Typical nearest-neighbour distance among training rows (leave-one-out),
    # estimated on a spread of probes. Floored so duplicate flag sets do not
    # make the ratio blow up.
    typical = 1.0
    if inputs and n > 0:
        step = max(1, n // _PROBES)
        probe_ids = list(range(0, n, step))[:_PROBES]
        nn_train = []
        for p in probe_ids:
            best_d = math.inf
            for j in range(n):
                if j != p:
                    d = dist(tl[p], tl[j])
                    if d < best_d:
                        best_d = d
            if best_d < math.inf:
                nn_train.append(best_d)
        floor = 0.5 / len(inputs)
        typical = max(_median(nn_train) if nn_train else 0.0, floor)

    out = []
    for r in rows:
        total = goal_part(r)
        if inputs and n > 0:
            rl = {c: _level(r.get(c)) for c in inputs}
            nn = min(dist(rl, t) for t in tl)
            total += nn / typical
        out.append(float(total))
    return out

