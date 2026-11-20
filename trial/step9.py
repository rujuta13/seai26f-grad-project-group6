"""Step 9: spot. Suspicion scores for benchmark rows that look like glitches or
flag sets unlike anything in train_csv.

odd(train_csv, rows) returns one non-negative score per row (higher = more
suspicious). The score adds up these checks; ordinary rows score near 0:
  - Unusable cells: a missing or non-numeric goal or numeric input cell, or a
    text input value that never appears in train_csv.
  - Input novelty: distance to the nearest training row in encoded input space,
    beyond the 95th percentile of training leave-one-out distances (as in step4).
  - Input range: a numeric input outside the training min..max, in training spans.
  - Goal disagreement: a goal value far from the goal values of the nearest
    training rows, in units of their spread. The spread is robust, so one bad
    training row does not hide a new glitch.
  - Goal outlier: a goal value far from the training median in robust (MAD)
    units, which catches typos such as an extra digit.
"""
import math

from step1 import _load
from step2 import _column_kinds, _medians, _vector

K_NEAR = 15        # nearest training rows used for the goal-disagreement check
CALIB = 200        # training rows used to calibrate input novelty
NOVEL_Q = 0.95     # quantile of training leave-one-out distances defining "normal"
RESID_OK = 3.0     # goal disagreement, in neighbour-spread units, treated as normal
ZOUT_OK = 6.0      # robust z-score of a goal treated as normal
CAP_SPAN = 10.0    # cap on a single range excess, in training spans
W_MISSING = 10.0   # penalty for a missing or non-numeric goal cell
W_BAD_INPUT = 2.0  # penalty for a missing or non-numeric numeric input cell
W_UNSEEN = 2.0     # penalty per text input whose value never appears in training
W_RANGE = 1.0      # weight per training span a numeric input lies beyond the range
EPS = 1e-12


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _quantile(vals, q):
    """Linear-interpolated quantile of a non-empty ascending list."""
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _median(vals):
    """Median of a non-empty ascending list."""
    return _quantile(vals, 0.5)


def _wmedian(pairs):
    """Weighted median of (value, weight) pairs with positive weights."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def _sq(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _nearest_sq(x, pool, skip=-1):
    best = math.inf
    for j, y in enumerate(pool):
        if j != skip:
            d = _sq(x, y)
            if d < best:
                best = d
    return best


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    train_rows, inputs, goals = _load(train_csv)
    rows = list(rows)
    m = len(rows)
    if m == 0:
        return []
    n = len(train_rows)
    names = [c for c, _ in goals]
    scores = [0.0] * m

    kinds, cats = _column_kinds(train_rows, inputs)
    cat_sets = {c: set(cats[c]) for c in cats}

    # 1. Unusable cells, out-of-range numbers, and text values never seen in training.
    ranges = {}
    for c in inputs:
        if kinds[c] == "num":
            vals = [x for x in (_finite(r.get(c)) for r in train_rows) if x is not None]
            if vals:
                ranges[c] = (min(vals), max(vals))

    for i, r in enumerate(rows):
        s = 0.0
        for c in inputs:
            v = r.get(c)
            if kinds[c] == "num":
                x = _finite(v)
                if x is None:
                    s += W_BAD_INPUT
                elif c in ranges:
                    lo, hi = ranges[c]
                    span = hi - lo
                    scale = span if span > 0 else max(abs(lo), 1.0)
                    beyond = max(lo - x, x - hi, 0.0)
                    if beyond > 0:
                        s += W_RANGE * min(CAP_SPAN, beyond / scale)
            elif _text(v) not in cat_sets[c]:
                s += W_UNSEEN
        scores[i] += s

    # 2. Input novelty, calibrated on training leave-one-out distances.
    has_x = bool(inputs) and n >= 2
    D = []
    if has_x:
        med = _medians(train_rows, inputs, kinds)
        X = [_vector(r, inputs, kinds, cats, med) for r in train_rows]
        Q = [_vector(r, inputs, kinds, cats, med) for r in rows]
        D = [[_sq(q, x) for x in X] for q in Q]

        step = max(1, n // CALIB)
        loo = sorted(_nearest_sq(X[i], X, skip=i) for i in range(0, n, step))
        thresh = loo[min(len(loo) - 1, int(NOVEL_Q * (len(loo) - 1)))]
        nscale = math.sqrt(thresh) if thresh > 0 else 1.0
        for i in range(m):
            nd = math.sqrt(min(D[i]))
            scores[i] += max(0.0, nd / nscale - 1.0)

    # 3. Goal checks: missing cells, outliers against the training median,
    #    and disagreement with the goals of the nearest training rows.
    for c in names:
        tr_idx, tr_y = [], []
        for k, r in enumerate(train_rows):
            y = _finite(r.get(c))
            if y is not None:
                tr_idx.append(k)
                tr_y.append(y)

        for i, r in enumerate(rows):
            if _finite(r.get(c)) is None:
                scores[i] += W_MISSING

        if not tr_y:
            continue

        gmed = _median(sorted(tr_y))
        mad = _median(sorted(abs(y - gmed) for y in tr_y))
        rsd = 1.4826 * mad
        if rsd <= 0:
            mu = sum(tr_y) / len(tr_y)
            var = sum((y - mu) ** 2 for y in tr_y) / max(1, len(tr_y) - 1)
            rsd = math.sqrt(var)

        for i, r in enumerate(rows):
            y = _finite(r.get(c))
            if y is None:
                continue

            dev = abs(y - gmed)
            if rsd > 0:
                scores[i] += max(0.0, dev / rsd - ZOUT_OK)
            elif dev > EPS * max(1.0, abs(gmed)):
                scores[i] += CAP_SPAN

            if has_x:
                cand = sorted((D[i][k], yv) for k, yv in zip(tr_idx, tr_y))[:K_NEAR]
                est = _wmedian([(yv, 1.0 / (1.0 + dv)) for dv, yv in cand])
                srt = sorted(yv for _, yv in cand)
                iqr = _quantile(srt, 0.75) - _quantile(srt, 0.25)
                spread = max(iqr, 0.25 * rsd, 0.01 * abs(est), EPS)
                scores[i] += max(0.0, abs(y - est) / spread - RESID_OK)

    return [float(s) for s in scores]

