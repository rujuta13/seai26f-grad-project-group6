import csv
import math
import random


def _to_num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def _load(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    goals = []   # (column name, sign): +1 maximise, -1 minimise
    inputs = []
    for c in cols:
        if c.endswith("+"):
            goals.append((c, 1.0))
        elif c.endswith("-"):
            goals.append((c, -1.0))
        elif c.endswith("X"):
            continue
        else:
            inputs.append(c)
    return rows, inputs, goals


def _features(rows, inputs):
    feats = [[0.0] * len(inputs) for _ in rows]
    for j, c in enumerate(inputs):
        raw = [r.get(c, "") for r in rows]
        nums = [_to_num(v) for v in raw]
        if all(x is not None for x in nums):
            for i, x in enumerate(nums):
                feats[i][j] = x
        else:
            codes = {}
            for i, v in enumerate(raw):
                feats[i][j] = float(codes.setdefault(str(v), len(codes)))
    return feats


def _sq(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _kmeans(X, k, seed=0, iters=100):
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])
    rng = random.Random(seed)

    # k-means++ initialisation (deterministic via seeded RNG)
    centers = [list(X[rng.randrange(n)])]
    dist2 = [_sq(x, centers[0]) for x in X]
    while len(centers) < k:
        total = sum(dist2)
        if total <= 0:
            idx = rng.randrange(n)
        else:
            r = rng.random() * total
            acc = 0.0
            idx = n - 1
            for i, w in enumerate(dist2):
                acc += w
                if acc >= r:
                    idx = i
                    break
        centers.append(list(X[idx]))
        dist2 = [min(dist2[i], _sq(X[i], centers[-1])) for i in range(n)]

    labels = [-1] * n
    for _ in range(iters):
        changed = False
        for i, x in enumerate(X):
            c_best = min(range(k), key=lambda c: _sq(x, centers[c]))
            if c_best != labels[i]:
                labels[i] = c_best
                changed = True
        if not changed and _ > 0:
            break

        # update centres
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for i, x in enumerate(X):
            c = labels[i]
            counts[c] += 1
            row = sums[c]
            for j in range(d):
                row[j] += x[j]
        for c in range(k):
            if counts[c] > 0:
                centers[c] = [s / counts[c] for s in sums[c]]
            else:
                # reseed empty cluster with the point farthest from its centre
                far = max(range(n), key=lambda i: _sq(X[i], centers[labels[i]]))
                centers[c] = list(X[far])
                labels[far] = c

    return labels


def families(train_csv) -> list[int]:
    rows, inputs, _goals = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    X = _features(rows, inputs)
    if not inputs:
        return [0] * n
    k = min(5, n)
    return [int(c) for c in _kmeans(X, k)]


def _dominates(a, b):
    strictly = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strictly = True
    return strictly


def best(train_csv) -> list[int]:
    rows, _inputs, goals = _load(train_csv)
    n = len(rows)
    cap = int(0.05 * n)
    if n == 0 or cap == 0 or not goals:
        return []

    # scores: higher is better on every goal; missing values count as worst
    scores = []
    for r in rows:
        s = []
        for name, sign in goals:
            v = _to_num(r.get(name, ""))
            s.append(sign * v if v is not None else -math.inf)
        scores.append(tuple(s))

    # Pareto front: rows not dominated by any other row
    front = [
        i for i in range(n)
        if not any(_dominates(scores[j], scores[i]) for j in range(n) if j != i)
    ]

    if len(front) > cap:
        # rank sums across goals (global), higher is better
        g = len(goals)
        rank_sum = [0] * n
        for gi in range(g):
            order = sorted(range(n), key=lambda i: scores[i][gi])
            for pos, i in enumerate(order):
                rank_sum[i] += pos
        front.sort(key=lambda i: (-rank_sum[i], i))
        front = front[:cap]

    return sorted(front)

