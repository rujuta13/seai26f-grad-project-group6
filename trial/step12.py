import math

import step1
import step2

_TAIL = 0.05  # interval covers the 5th to 95th weighted percentile of neighbours


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate each goal of `row` from its nearest training rows.

    `row` carries only input columns, so goals are never read from it. Each
    goal gets a weighted median of the nearest training rows' values as the
    estimate, and a 5th-95th percentile range as the band the true value
    should fall in. If no training row has a finite value for a goal, the
    estimate is 0.0 with an unbounded range.
    """
    rows, inputs, goals = step1._load(train_csv)
    if not goals:
        return {}
    n = len(rows)
    if n == 0:
        return {name: (0.0, -math.inf, math.inf) for name, _ in goals}

    kinds = step2._prep(rows, inputs)
    train_parsed = [step2._parse(r, inputs, kinds) for r in rows]
    q = step2._parse(row, inputs, kinds)

    dists = sorted(
        ((step2._dist(q, t, kinds), i) for i, t in enumerate(train_parsed)),
        key=lambda p: (p[0], p[1]),
    )
    nearest = dists[: min(step2.K, n)]

    result = {}
    for name, _sign in goals:
        pairs = []
        for d, i in nearest:
            v = step1._to_num(rows[i].get(name, ""))
            if v is not None and math.isfinite(v):
                pairs.append((v, 1.0 / (0.1 + d)))

        # fallback: all training rows with a finite value, equal weight
        if not pairs:
            for r in rows:
                v = step1._to_num(r.get(name, ""))
                if v is not None and math.isfinite(v):
                    pairs.append((v, 1.0))

        if not pairs:
            result[name] = (0.0, -math.inf, math.inf)
            continue

        pairs.sort(key=lambda p: p[0])
        est = step2._wquantile(pairs, 0.5)
        lo = min(step2._wquantile(pairs, _TAIL), est)
        hi = max(step2._wquantile(pairs, 1.0 - _TAIL), est)
        result[name] = (float(est), float(lo), float(hi))

    return result

