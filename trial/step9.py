"""Step 9: flag benchmark results that look like glitches or unusual flag sets.

odd() returns one suspicion score per row (higher = more suspicious). Two signals
are combined, both measured against train_csv:

* goal residual: for each goal, how far the reported value is from the median
  goal value of the K training builds with the most similar flags, in robust
  units (the goal's median absolute deviation over the whole training set). The
  largest residual over the goals is used, so one absurd goal value is enough to
  raise the score. A missing or non-finite goal scores MISSING_RESIDUAL.
* novelty: the mean normalised flag distance to the K nearest training builds,
  the same distance step2 and step4 use. High novelty means the flag set is
  unlike anything in train_csv.

score = largest goal residual + NOVELTY_WEIGHT * novelty

Scores are plain floats on a relative scale; compare them with each other rather
than against a fixed cut-off.
"""

import heapq
import math

from step1 import _load, _split_columns

K = 15
NOVELTY_WEIGHT = 5.0
MISSING_RESIDUAL = 50.0
MAD_TO_SIGMA = 1.4826


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _median(sorted_vals):
    n = len(sorted_vals)
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _scale(vals):
    """Robust spread of a goal: MAD-based, else std, else a tiny positive value."""
    med = _median(sorted(vals))
    mad = _median(sorted(abs(v - med) for v in vals)) * MAD_TO_SIGMA
    if mad > 0.0:
        return mad
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    if sd > 0.0:
        return sd
    return max(abs(med) * 1e-3, 1e-9)


def _spec(rows, inputs):
    """Per input column: ('num', lo, hi) if every training value is numeric, else ('cat', None, None)."""
    spec = {}
    for c in inputs:
        vals = [r.get(c) for r in rows]
        if vals and all(isinstance(v, float) for v in vals):
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                spec[c] = ("num", min(finite), max(finite))
                continue
        spec[c] = ("cat", None, None)
    return spec


def _distance(a, b, inputs, spec):
    """Sum of per-column differences; numeric columns are scaled to [0, 1]."""
    d = 0.0
    for c in inputs:
        kind, lo, hi = spec[c]
        x = a.get(c)
        y = b.get(c)
        if kind == "num":
            fx = _finite(x)
            fy = _finite(y)
            if fx is None or fy is None:
                d += 1.0
            else:
                span = hi - lo
                d += abs(fx - fy) / span if span > 0 else 0.0
        else:
            d += 0.0 if str(x) == str(y) else 1.0
    return d


def odd(train_csv, rows) -> list[float]:
    """Return one suspicion score per row of `rows` (higher = more likely a glitch or unusual).

    Each score is the largest robust goal residual against the most similar training
    builds, plus NOVELTY_WEIGHT times the mean flag-space distance to those builds.
    If train_csv has no rows there is no reference, so every score is 0.0.
    """
    columns, train = _load(train_csv)
    maximise, minimise, inputs = _split_columns(columns)
    goals = maximise + minimise

    if not rows:
        return []
    n = len(train)
    if n == 0:
        return [0.0] * len(rows)

    spec = _spec(train, inputs)
    width = max(1, len(inputs))
    k = min(K, n)

    train_goal = {g: [_finite(r.get(g)) for r in train] for g in goals}
    scale = {}
    for g in goals:
        vals = [v for v in train_goal[g] if v is not None]
        if vals:
            scale[g] = _scale(vals)

    out = []
    for row in rows:
        near = heapq.nsmallest(
            k,
            ((_distance(t, row, inputs, spec), i) for i, t in enumerate(train)),
        )
        novelty = sum(d for d, _ in near) / (k * width)

        worst = 0.0
        for g in goals:
            if g not in scale:
                continue
            v = _finite(row.get(g))
            if v is None:
                worst = max(worst, MISSING_RESIDUAL)
                continue
            local = sorted(
                train_goal[g][i] for _, i in near if train_goal[g][i] is not None
            )
            if not local:
                continue
            residual = abs(v - _median(local)) / scale[g]
            worst = max(worst, residual)

        out.append(float(worst + NOVELTY_WEIGHT * novelty))
    return out

