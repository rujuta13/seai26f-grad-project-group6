import math

import step1

K = 20  # number of nearest training rows used for the guess


def _prep(rows, inputs):
    """Decide per input column whether it is numeric or categorical."""
    kinds = []
    for c in inputs:
        nums = [step1._to_num(r.get(c, "")) for r in rows]
        if all(x is not None for x in nums):
            lo, hi = min(nums), max(nums)
            span = (hi - lo) if hi > lo else 1.0
            kinds.append(("num", span))
        else:
            kinds.append(("cat", None))
    return kinds


def _parse(row, inputs, kinds):
    out = []
    for c, (kind, _span) in zip(inputs, kinds):
        v = row.get(c, "")
        if kind == "num":
            out.append(step1._to_num(v))  # float or None
        else:
            out.append(str(v))
    return out


def _dist(a, b, kinds):
    d = 0.0
    for (kind, span), x, y in zip(kinds, a, b):
        if kind == "num":
            if x is None or y is None:
                d += 1.0
            else:
                t = (x - y) / span
                d += t * t
        else:
            if x != y:
                d += 1.0
    return math.sqrt(d)


def _wquantile(pairs, q):
    """pairs: list of (value, weight) sorted by value."""
    total = sum(w for _, w in pairs)
    target = q * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            return v
    return pairs[-1][0]


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    rows, inputs, goals = step1._load(train_csv)
    if not goals:
        return {}
    n = len(rows)
    if n == 0:
        return {name: (0.0, -math.inf, math.inf) for name, _ in goals}

    kinds = _prep(rows, inputs)
    train_parsed = [_parse(r, inputs, kinds) for r in rows]
    q = _parse(row, inputs, kinds)

    dists = sorted(
        ((_dist(q, t, kinds), i) for i, t in enumerate(train_parsed)),
        key=lambda p: (p[0], p[1]),
    )
    nearest = dists[: min(K, n)]

    result = {}
    for name, _sign in goals:
        # weighted values from nearest neighbours that have this goal
        pairs = []
        for d, i in nearest:
            v = step1._to_num(rows[i].get(name, ""))
            if v is not None:
                pairs.append((v, 1.0 / (0.1 + d)))

        # fallback: all training rows, equal weight
        if not pairs:
            for r in rows:
                v = step1._to_num(r.get(name, ""))
                if v is not None:
                    pairs.append((v, 1.0))

        if not pairs:
            result[name] = (0.0, -math.inf, math.inf)
            continue

        pairs.sort(key=lambda p: p[0])
        est = _wquantile(pairs, 0.5)
        lo = min(_wquantile(pairs, 0.1), est)
        hi = max(_wquantile(pairs, 0.9), est)
        result[name] = (est, lo, hi)

    return result

