import math

import step1


def _column_values(rows, c):
    """Return ("num", floats) if every cell is finite, else ("cat", strings)."""
    raw = [r.get(c, "") for r in rows]
    nums = [step1._to_num(v) for v in raw]
    if all(x is not None and math.isfinite(x) for x in nums):
        return "num", nums
    return "cat", [str(v) for v in raw]


def _numeric_effect(bad_vals, rest_vals):
    """Absolute standardised mean difference between the two groups."""
    nb, nr = len(bad_vals), len(rest_vals)
    mb = sum(bad_vals) / nb
    mr = sum(rest_vals) / nr
    diff = abs(mb - mr)
    vb = sum((x - mb) ** 2 for x in bad_vals) / (nb - 1) if nb > 1 else 0.0
    vr = sum((x - mr) ** 2 for x in rest_vals) / (nr - 1) if nr > 1 else 0.0
    pooled = math.sqrt(((nb - 1) * vb + (nr - 1) * vr) / (nb + nr - 2))
    if pooled > 1e-12:
        return diff / pooled
    # no spread inside the groups: any difference in means is complete separation
    return 0.0 if diff <= 1e-12 else math.inf


def _categorical_effect(bad_vals, rest_vals):
    """Largest absolute difference in value frequency between the two groups."""
    nb, nr = len(bad_vals), len(rest_vals)
    cb = {}
    for v in bad_vals:
        cb[v] = cb.get(v, 0) + 1
    cr = {}
    for v in rest_vals:
        cr[v] = cr.get(v, 0) + 1
    best = 0.0
    for v in set(cb) | set(cr):
        d = abs(cb.get(v, 0) / nb - cr.get(v, 0) / nr)
        if d > best:
            best = d
    return best


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Bad rows are the worst quarter of the training rows, ranked by the
    sum of their per-goal ranks (missing values count as worst). Each input
    is scored by how strongly its values separate bad rows from the rest.
    Inputs that separate the bad rows most come first. Ties keep column order.
    """
    rows, inputs, goals = step1._load(train_csv)
    n = len(rows)
    if n < 4 or not inputs or not goals:
        return list(inputs)

    # rank sum across goals: low means bad (missing values count as worst)
    rank_sum = [0] * n
    for name, sign in goals:
        scores = []
        for r in rows:
            v = step1._to_num(r.get(name, ""))
            scores.append(sign * v if v is not None else -math.inf)
        order = sorted(range(n), key=lambda i: (scores[i], i))
        for pos, i in enumerate(order):
            rank_sum[i] += pos

    order = sorted(range(n), key=lambda i: (rank_sum[i], i))
    k = max(1, n // 4)
    bad_set = set(order[:k])
    bad_rows = [i for i in range(n) if i in bad_set]
    rest_rows = [i for i in range(n) if i not in bad_set]

    scored = []
    for j, c in enumerate(inputs):
        kind, vals = _column_values(rows, c)
        b = [vals[i] for i in bad_rows]
        r_ = [vals[i] for i in rest_rows]
        if kind == "num":
            s = _numeric_effect(b, r_)
        else:
            s = _categorical_effect(b, r_)
        scored.append((-s, j, c))

    scored.sort()
    return [c for _, _, c in scored]

