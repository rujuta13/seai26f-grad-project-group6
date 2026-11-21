import math
from collections import Counter

import step1

# Bad rows are the worst quarter of the training rows (by rank sum across goals).
_BAD_DIVISOR = 4


def _column(rows, c):
    """("num", floats) if every cell is a finite number, else ("cat", strings)."""
    raw = [r.get(c, "") for r in rows]
    nums = [step1._to_num(v) for v in raw]
    if all(x is not None and math.isfinite(x) for x in nums):
        return "num", nums
    return "cat", [str(v) for v in raw]


def _numeric_effect(bad, rest):
    """Absolute standardised mean difference between bad rows and the rest."""
    nb, nr = len(bad), len(rest)
    mb = sum(bad) / nb
    mr = sum(rest) / nr
    diff = abs(mb - mr)
    vb = sum((x - mb) ** 2 for x in bad) / (nb - 1) if nb > 1 else 0.0
    vr = sum((x - mr) ** 2 for x in rest) / (nr - 1) if nr > 1 else 0.0
    pooled_var = ((nb - 1) * vb + (nr - 1) * vr) / (nb + nr - 2)
    pooled = math.sqrt(max(0.0, pooled_var))
    if pooled > 1e-12:
        return diff / pooled
    # no spread inside the groups: any gap in means is complete separation
    return 0.0 if diff <= 1e-12 else math.inf


def _categorical_effect(bad, rest):
    """Largest absolute difference in value frequency between the two groups."""
    nb, nr = len(bad), len(rest)
    cb = Counter(bad)
    cr = Counter(rest)
    return max(
        (abs(cb.get(v, 0) / nb - cr.get(v, 0) / nr) for v in set(cb) | set(cr)),
        default=0.0,
    )


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Bad rows are the worst quarter of the training rows, ranked by the sum of
    their per-goal ranks (missing values count as worst). Every input is scored
    by how strongly its values separate bad rows from the rest; inputs that
    separate them most come first. Ties keep column order.

    The goals are not named in the signature, so all goal columns count
    toward "bad". A goal-specific question (for example, weakness from the
    penalty spot) is answered by the rank of the inputs this returns, not by a
    separate goal filter.
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
    k = max(1, n // _BAD_DIVISOR)
    bad_set = set(order[:k])
    bad_rows = [i for i in range(n) if i in bad_set]
    rest_rows = [i for i in range(n) if i not in bad_set]

    scored = []
    for j, c in enumerate(inputs):
        kind, vals = _column(rows, c)
        bad = [vals[i] for i in bad_rows]
        rest = [vals[i] for i in rest_rows]
        if kind == "num":
            s = _numeric_effect(bad, rest)
        else:
            s = _categorical_effect(bad, rest)
        scored.append((-s, j, c))

    scored.sort()
    return [c for _, _, c in scored]

