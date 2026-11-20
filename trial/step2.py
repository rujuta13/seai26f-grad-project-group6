"""step2.py: guess how a flag set will perform before it is benchmarked.

The guess is a k-nearest-neighbour estimate over the training rows. Reading,
column splitting, feature scaling, distances and number parsing come from step1.
"""
from __future__ import annotations

import math
from collections import Counter

from step1 import _columns, _dist2, _features, _num, _read

K_NEIGHBOURS = 7        # training rows that vote on each guess
NOISE_FLOOR = 0.05      # benchmark noise assumed even when neighbours agree


def _quantile(vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of a non-empty list of floats."""
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _fill(row: dict, rows: list[dict], inputs: list[str]) -> dict:
    """Copy the input cells of row. Missing or unusable cells are replaced by
    the training median (numeric columns) or the most common training value
    (other columns)."""
    filled = {}
    for c in inputs:
        numeric = all(isinstance(r.get(c), float) for r in rows)
        v = row.get(c)
        if numeric:
            if _num(v) is not None:
                filled[c] = v
            else:
                vals = [x for x in (_num(r.get(c)) for r in rows) if x is not None]
                filled[c] = _quantile(vals, 0.5) if vals else 0.0
        else:
            if isinstance(v, str) and v != "":
                filled[c] = v
            else:
                common = Counter(
                    r.get(c) for r in rows
                    if isinstance(r.get(c), str) and r.get(c) != ""
                )
                filled[c] = common.most_common(1)[0][0] if common else ""
    return filled


def guess(train_csv, row: dict) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a flag set described by row (goal columns removed).

    Returns {goal: (estimate, low, high)}. The estimate is the mean goal value
    of the nearest training rows (among rows with a usable value for that goal).
    low..high is roughly the 10th-90th percentile of those neighbours' values,
    widened by a benchmark-noise floor and by how far the flag set lies from
    the training rows (features are scaled to [0, 1]). Far-away flag sets get
    wider intervals. If nothing usable is known, the estimate is NaN and the
    interval is unbounded.
    """
    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    unknown = (math.nan, -math.inf, math.inf)
    if n == 0 or not goals:
        return {g: unknown for g in goals}

    feats = _features(rows + [_fill(row, rows, inputs)], inputs)
    query, train = feats[-1], feats[:-1]
    d2 = [_dist2(f, query) for f in train]

    out: dict[str, tuple[float, float, float]] = {}
    for g in goals:
        cand = [i for i in range(n) if _num(rows[i].get(g)) is not None]
        if not cand:
            out[g] = unknown
            continue
        cand.sort(key=lambda i: (d2[i], i))
        near = cand[:K_NEIGHBOURS]
        vals = [rows[i][g] for i in near]

        est = sum(vals) / len(vals)
        q10 = _quantile(vals, 0.1)
        q90 = _quantile(vals, 0.9)
        far = min(1.0, math.sqrt(sum(d2[i] for i in near) / len(near)))

        base = max((q90 - q10) / 2, NOISE_FLOOR * abs(est))
        pad = base + 0.5 * far * abs(est)
        low = min(q10 - pad, est)
        high = max(q90 + pad, est)
        out[g] = (est, low, high)
    return out

