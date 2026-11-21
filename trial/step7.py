import math
import random

import step1
import step2

_SEED = 0  # fixed seed so the same inputs always give the same answer


def _resolve(goals, goal):
    """Map the `goal` argument to the one matching goal column."""
    exact = [(n, s) for n, s in goals if n == goal]
    if exact:
        return exact
    key = goal.rstrip("+-").lower()
    loose = [(n, s) for n, s in goals if n.rstrip("+-").lower() == key]
    if len(loose) == 1:
        return loose
    names = ", ".join(n for n, _ in goals)
    raise ValueError(f"goal {goal!r} does not name one goal column (goals: {names})")


def _pick(cache, active):
    """Best labelled row: undominated first, then highest rank sum, then lowest index."""
    idxs = sorted(cache)
    if len(active) == 1:
        return max(idxs, key=lambda i: (cache[i][0], -i))
    scores = [cache[i] for i in idxs]
    m = len(idxs)
    front = [
        k for k in range(m)
        if not any(step1._dominates(scores[j], scores[k])
                   for j in range(m) if j != k)
    ]
    rank = [0] * m
    for g in range(len(active)):
        order = sorted(range(m), key=lambda k: (scores[k][g], idxs[k]))
        for pos, k in enumerate(order):
            rank[k] += pos
    best_k = max(front, key=lambda k: (rank[k], -idxs[k]))
    return idxs[best_k]


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, labelling at most `budget` rows.

    Rows are labelled by alternating exploration (farthest from every row
    labelled so far, after an initial exploration half of the budget) with
    exploitation (the unlabelled row nearest to the current best). If `goal`
    names one goal column, only that goal is optimised. Deterministic via a
    fixed seed.
    """
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")

    rows, inputs, goals = step1._load(blind_csv)
    if goal is not None:
        goals = _resolve(goals, goal)
    if not goals:
        raise ValueError("file has no goal columns to optimise")
    n = len(rows)
    if n == 0:
        raise ValueError("file has no rows")

    kinds = step2._prep(rows, inputs)
    parsed = [step2._parse(r, inputs, kinds) for r in rows]

    cache = {}  # row index -> goal tuple, signed so higher is better

    def measure(i):
        if i in cache:
            return
        if len(cache) >= budget:
            raise RuntimeError(f"label budget of {budget} exhausted")
        got = label(i)
        vals = []
        for name, sign in goals:
            v = step1._to_num(got.get(name))
            vals.append(sign * v if v is not None else -math.inf)
        cache[i] = tuple(vals)

    def dist(i, j):
        return step2._dist(parsed[i], parsed[j], kinds)

    m0 = max(1, budget // 2)  # evaluations spent on initial exploration
    rng = random.Random(_SEED)
    mind = [math.inf] * n     # distance from each row to the labelled set

    while len(cache) < min(budget, n):
        unl = [i for i in range(n) if i not in cache]
        if not cache:
            nxt = rng.randrange(n)
        elif len(cache) < m0 or len(cache) % 2 == 1:
            # explore: the unlabelled row farthest from everything labelled
            nxt = max(unl, key=lambda i: (mind[i], -i))
        else:
            # exploit: the unlabelled row nearest the current best
            inc = _pick(cache, goals)
            nxt = min(unl, key=lambda i: (dist(i, inc), i))
        measure(nxt)
        for i in range(n):
            d = dist(i, nxt)
            if d < mind[i]:
                mind[i] = d

    return int(_pick(cache, goals))

