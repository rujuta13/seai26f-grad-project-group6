import math
import random

import step1

_SEED = 0          # fixed seed so the same inputs always give the same answer
_REF_CAP = 300     # max training rows used as leave-one-out reference
_MISSING = 3.0     # penalty for a missing or non-finite goal or input cell
_UNSEEN = 1.0      # penalty for a categorical value never seen in training
_Z_FREE = 3.0      # goal z-scores up to this are treated as normal
_CAP = 10.0        # no single component contributes more than this


def _finite(v):
    x = step1._to_num(v)
    if x is None or not math.isfinite(x):
        return None
    return x


def _median(xs):
    s = sorted(xs)
    m = len(s)
    if m == 0:
        return 0.0
    mid = m // 2
    if m % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _quantile(xs, q):
    s = sorted(xs)
    k = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[k]


def _robust_scale(xs, med):
    """Spread estimate: scaled MAD, falling back to standard deviation."""
    mad = _median([abs(x - med) for x in xs])
    if mad > 0:
        return 1.4826 * mad
    m = len(xs)
    var = sum((x - med) ** 2 for x in xs) / m if m else 0.0
    return math.sqrt(var)


def _goal_excess(v, med, scale):
    """How far a goal value sits outside the training bulk."""
    if scale > 0:
        z = abs(v - med) / scale
    else:
        # constant goal in training: any different value is unusual
        z = 0.0 if v == med else _Z_FREE + 1.0
    return min(_CAP, max(0.0, z - _Z_FREE))


def _prep_inputs(trows, inputs):
    spec = []
    for c in inputs:
        nums = [_finite(r.get(c, "")) for r in trows]
        if nums and all(x is not None for x in nums):
            lo, hi = min(nums), max(nums)
            span = hi - lo
            med = _median(nums)
            scale = span if span > 0 else max(1.0, abs(med))
            spec.append({"kind": "num", "lo": lo, "hi": hi,
                         "span": span, "scale": scale})
        else:
            seen = {str(r.get(c, "")) for r in trows}
            spec.append({"kind": "cat", "seen": seen})
    return spec


def _parse(row, inputs, spec):
    out = []
    for c, s in zip(inputs, spec):
        raw = row.get(c, "")
        if s["kind"] == "num":
            out.append(_finite(raw))
        else:
            out.append(str(raw))
    return out


def _dist(a, b, spec):
    d = 0.0
    for s, x, y in zip(spec, a, b):
        if s["kind"] == "num":
            if x is None or y is None:
                d += 1.0
            else:
                t = (x - y) / s["scale"]
                d += t * t
        elif x != y:
            d += 1.0
    return math.sqrt(d)


def _range_excess(x, s):
    """How far a numeric input lies outside the training range, in scale units."""
    beyond = max(0.0, s["lo"] - x, x - s["hi"]) / s["scale"]
    return min(_CAP, beyond)


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row (goals included).

    Higher means more likely a glitch, a typo, or unlike anything in
    train_csv. Each row collects penalties for missing or non-finite cells,
    unseen categorical values, numeric inputs outside the training range,
    goal values far from the training bulk (robust z-score), and an input
    configuration that is far from every training row compared with how far
    training rows are from each other. Deterministic via a fixed seed.
    """
    rows = list(rows)
    m = len(rows)
    if m == 0:
        return []

    trows, inputs, goals = step1._load(train_csv)
    if not trows:
        return [0.0] * m

    scores = [0.0] * m

    spec = _prep_inputs(trows, inputs)
    tparsed = [_parse(r, inputs, spec) for r in trows]
    qparsed = [_parse(r, inputs, spec) for r in rows]

    # per-input checks: typos, out-of-range numbers, unseen categories
    for k, s in enumerate(spec):
        for i in range(m):
            x = qparsed[i][k]
            if s["kind"] == "num":
                if x is None:
                    scores[i] += _MISSING
                else:
                    scores[i] += _range_excess(x, s)
            else:
                if x not in s["seen"]:
                    scores[i] += _UNSEEN

    # joint novelty: distance to nearest training row, relative to how far
    # training rows usually are from their own nearest neighbour
    n = len(tparsed)
    if n >= 2 and inputs:
        idx = list(range(n))
        if n > _REF_CAP:
            idx = random.Random(_SEED).sample(idx, _REF_CAP)
        ref = []
        for i in idx:
            ref.append(min(_dist(tparsed[i], tparsed[j], spec)
                           for j in range(n) if j != i))
        bound = _quantile(ref, 0.95)
        for i in range(m):
            nn = min(_dist(qparsed[i], t, spec) for t in tparsed)
            scores[i] += min(_CAP, max(0.0, nn - bound) / max(bound, 1.0))

    # goal checks: values far from the training bulk, or missing
    for name, _sign in goals:
        tv = [x for x in (_finite(r.get(name, "")) for r in trows) if x is not None]
        if not tv:
            continue
        med = _median(tv)
        scale = _robust_scale(tv, med)
        for i, r in enumerate(rows):
            v = _finite(r.get(name, ""))
            if v is None:
                scores[i] += _MISSING
            else:
                scores[i] += _goal_excess(v, med, scale)

    return [float(s) for s in scores]

