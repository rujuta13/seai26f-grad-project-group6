import math

from step1 import _load

NEIGHBOURS = 15   # how many nearest training rows inform the guess
TAIL = 0.10       # interval core is the 10th..90th percentile of neighbour values


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _column_kinds(rows, inputs):
    """Mirror step1's encoding: numeric columns stay numeric, others are one-hot."""
    kinds, cats = {}, {}
    for c in inputs:
        raw = [r[c] for r in rows]
        try:
            for v in raw:
                float(v)
            kinds[c] = "num"
        except (TypeError, ValueError):
            kinds[c] = "cat"
            cats[c] = sorted({_text(v) for v in raw})
    return kinds, cats


def _medians(rows, inputs, kinds):
    med = {}
    for c in inputs:
        if kinds[c] == "num":
            vals = sorted(float(r[c]) for r in rows)
            med[c] = vals[len(vals) // 2] if vals else 0.0
    return med


def _vector(values, inputs, kinds, cats, med):
    out = []
    for c in inputs:
        x = values.get(c)
        if kinds[c] == "num":
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                out.append(med[c])   # unusable query cell: fall back to training median
        else:
            s = _text(x)
            out.extend(1.0 if s == cat else 0.0 for cat in cats[c])
    return out


def _quantile(vals, q):
    """Linear-interpolated quantile of a non-empty ascending list."""
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _weighted_median(pairs):
    """pairs: list of (value, weight) with positive weights."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a flag set `row` (goal columns already removed).

    Method: nearest-neighbour regression over the training rows.
      - estimate: distance-weighted median of the nearest training rows' goal values
      - core interval: 10th..90th percentile of those neighbours' values
      - widened by a margin that grows when the query is unlike its neighbours
        (novelty) and when the goal varies a lot across the whole training set.
    Returns {goal: (estimate, low, high)} with low <= estimate <= high.
    """
    rows, inputs, goals = _load(train_csv)
    names = [c for c, _ in goals]
    n = len(rows)
    if not names:
        return {}
    if n == 0:
        return {c: (0.0, 0.0, 0.0) for c in names}

    kinds, cats = _column_kinds(rows, inputs)
    med = _medians(rows, inputs, kinds)
    X = [_vector(r, inputs, kinds, cats, med) for r in rows]
    q = _vector(row, inputs, kinds, cats, med)
    F = len(q)

    if F:
        d = [sum((a - b) * (a - b) for a, b in zip(x, q)) for x in X]
        order = sorted(range(n), key=lambda i: (d[i], i))
        nb = order[:min(NEIGHBOURS, n)]
        w = {i: 1.0 / (1.0 + d[i]) for i in nb}
        mean_d = sum(d[i] for i in nb) / len(nb)
        # Roughly the fraction of features on which the query differs from its neighbours.
        nov = min(1.0, math.sqrt(mean_d) / math.sqrt(F))
    else:
        # No usable inputs: know nothing about the row, use the whole training set, wide interval.
        nb = list(range(n))
        w = {i: 1.0 for i in nb}
        nov = 1.0

    out = {}
    for name in names:
        y = [float(r[name]) for r in rows]
        span = max(y) - min(y)

        est = _weighted_median([(y[i], w[i]) for i in nb])
        vals = sorted(y[i] for i in nb)
        lo_q = _quantile(vals, TAIL)
        hi_q = _quantile(vals, 1.0 - TAIL)

        margin = 0.02 * abs(est) + span * (0.05 + 0.25 * nov)
        low = min(lo_q, est) - margin
        high = max(hi_q, est) + margin
        out[name] = (float(est), float(low), float(high))
    return out

