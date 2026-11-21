import math

import step1
import step2


def _scores(row, goals):
    """Higher is better on every goal; missing values count as worst."""
    s = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        s.append(sign * v if v is not None else -math.inf)
    return tuple(s)


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row`, changed in as few values as possible so it becomes good.

    Only configurations present in train_csv are used, so the goals of a
    fixed row are known rather than guessed. Among training rows that no
    other training row dominates (the rows step6.why calls good) and that
    agree with `row` on every frozen column, the one differing from `row` in
    the fewest input values is chosen. Its values are copied into the
    returned dict. Ties go to the nearest such row, then the lowest index.

    If no undominated training row agrees with the frozen columns, the row
    with the fewest dominating rows is used instead, so the result may still
    be bad. If no training row agrees with the frozen columns, the inputs
    are returned unchanged.

    Columns named in `frozen` that are not inputs are ignored.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)

    rows, inputs, goals = step1._load(train_csv)
    out = {c: row.get(c, "") for c in inputs}
    if not rows or not inputs or not goals:
        return out

    kinds = step2._prep(rows, inputs)
    tparsed = [step2._parse(r, inputs, kinds) for r in rows]
    q = step2._parse(row, inputs, kinds)
    frozen_idx = [j for j, c in enumerate(inputs) if c in frozen]

    n = len(rows)
    scores = [_scores(r, goals) for r in rows]
    dom_count = [
        sum(1 for j in range(n)
            if j != i and step1._dominates(scores[j], scores[i]))
        for i in range(n)
    ]

    def compatible(i):
        return all(tparsed[i][j] == q[j] for j in frozen_idx)

    def changes(i):
        return sum(1 for j in range(len(inputs)) if tparsed[i][j] != q[j])

    def gap(i):
        return step2._dist(q, tparsed[i], kinds)

    compat = [i for i in range(n) if compatible(i)]
    if not compat:
        return out

    good = [i for i in compat if dom_count[i] == 0]
    if good:
        pick = min(good, key=lambda i: (changes(i), gap(i), i))
    else:
        pick = min(compat, key=lambda i: (dom_count[i], changes(i), gap(i), i))

    for j, c in enumerate(inputs):
        if tparsed[pick][j] != q[j]:
            raw = rows[pick].get(c, "")
            if kinds[j][0] == "num":
                out[c] = step1._to_num(raw)
            else:
                out[c] = str(raw)
    return out

