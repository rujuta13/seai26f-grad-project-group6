import math
import random

import step1
import step2
import step3

_WINDOW = 25      # new rows per drift test block (non-overlapping)
_REF_CAP = 200    # max training rows used as drift reference
_REF_SEED = 0     # fixed seed so the same inputs always give the same answer


def _scores(row, goals):
    """Higher is better on every goal; missing values count as worst."""
    s = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        s.append(sign * v if v is not None else -math.inf)
    return tuple(s)


def _insert(front, s):
    """Add s to a Pareto front (list of score tuples).

    Returns (new_front, dominated). If s is dominated by a member of the
    front, the front is returned unchanged. Dominance is transitive, so a
    point is dominated by some seen point iff it is dominated by the front.
    """
    for p in front:
        if step1._dominates(p, s):
            return front, True
    return [p for p in front if not step1._dominates(s, p)] + [s], False


def stream(train_csv, rows) -> list[bool]:
    """Keep a streamed row if it is not dominated on the goals by any
    training row or by any earlier streamed row.

    Only train_csv and the rows already seen are used for each decision.
    """
    trows, _inputs, goals = step1._load(train_csv)
    if not goals:
        return [False] * len(rows)

    front = []
    for r in trows:
        front, _ = _insert(front, _scores(r, goals))

    keep = []
    for r in rows:
        s = _scores(r, goals)
        front, dominated = _insert(front, s)
        keep.append(not dominated)
    return keep


def drift(train_csv, rows) -> int | None:
    """Index of the row at which new rows first look unlike train_csv, or None.

    Works on input columns only. Each non-overlapping block of _WINDOW new
    rows is summarised by its distance to the nearest training row. That
    sample is compared with the leave-one-out nearest-neighbour distances of
    the training rows, using a permutation test (step3.same). The first block
    that differs significantly is flagged, and its last row index is returned.
    Each decision uses only train_csv and rows up to that index.
    """
    trows, inputs, _goals = step1._load(train_csv)
    n = len(trows)
    if n < 2 or not rows or not inputs:
        return None

    kinds = step2._prep(trows, inputs)
    tparsed = [step2._parse(r, inputs, kinds) for r in trows]

    # reference: leave-one-out nearest-neighbour distances within training
    rng = random.Random(_REF_SEED)
    idx = list(range(n))
    if n > _REF_CAP:
        idx = rng.sample(idx, _REF_CAP)
    ref = []
    for i in idx:
        ref.append(min(
            step2._dist(tparsed[i], tparsed[j], kinds)
            for j in range(n) if j != i
        ))

    # stream the new rows in blocks
    qd = []
    for i, r in enumerate(rows):
        q = step2._parse(r, inputs, kinds)
        qd.append(min(step2._dist(q, t, kinds) for t in tparsed))
        if len(qd) == _WINDOW:
            if not step3.same(ref, qd):
                return i
            qd = []
    return None

