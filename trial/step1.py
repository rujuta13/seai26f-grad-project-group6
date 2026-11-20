import csv
import random

K = 5          # number of families reported
SEED = 1       # fixed seed so results are reproducible
MAX_ITER = 200


def _load(path):
    """Read a MOOT csv. Returns (rows, input_columns, goal_columns).
    goal_columns is a list of (name, sign): +1 maximise, -1 minimise."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    inputs, goals = [], []
    for c in cols:
        if c.endswith("+"):
            goals.append((c, 1))
        elif c.endswith("-"):
            goals.append((c, -1))
        elif c.endswith("X"):
            continue
        else:
            inputs.append(c)
    return rows, inputs, goals


def _encode(rows, inputs):
    """Turn input columns into one numeric vector per row.
    Numeric columns are used as-is; text columns are one-hot encoded."""
    X = [[] for _ in rows]
    for c in inputs:
        raw = [r[c] for r in rows]
        try:
            nums = [float(v) for v in raw]
            for i, v in enumerate(nums):
                X[i].append(v)
        except ValueError:
            cats = sorted(set(raw))
            for i, v in enumerate(raw):
                X[i].extend(1.0 if v == cat else 0.0 for cat in cats)
    return X


def _sq(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _kmeans(X, k, seed):
    """Plain k-means with k-means++ initialisation, standard library only."""
    n = len(X)
    d = len(X[0])
    rng = random.Random(seed)

    # k-means++ seeding
    centers = [list(X[rng.randrange(n)])]
    while len(centers) < k:
        dists = [min(_sq(x, c) for c in centers) for x in X]
        total = sum(dists)
        if total == 0:
            centers.append(list(X[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        chosen = n - 1
        for i, dv in enumerate(dists):
            acc += dv
            if acc >= r:
                chosen = i
                break
        centers.append(list(X[chosen]))

    labels = [-1] * n
    for _ in range(MAX_ITER):
        changed = False
        for i, x in enumerate(X):
            best_c = min(range(k), key=lambda c: _sq(x, centers[c]))
            if best_c != labels[i]:
                labels[i] = best_c
                changed = True
        if not changed:
            break
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for i, x in enumerate(X):
            c = labels[i]
            counts[c] += 1
            for j in range(d):
                sums[c][j] += x[j]
        for c in range(k):
            if counts[c]:
                centers[c] = [s / counts[c] for s in sums[c]]
            else:
                centers[c] = list(X[rng.randrange(n)])
    return labels


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    rows, inputs, _ = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    X = _encode(rows, inputs)
    if not X[0]:
        return [0] * n
    distinct = len({tuple(x) for x in X})
    k = max(1, min(K, distinct))
    return _kmeans(X, k, SEED)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows. Uses the Pareto front over the goals, ranked by
    mean normalised goal score."""
    rows, _, goals = _load(train_csv)
    n = len(rows)
    m = int(0.05 * n)
    if n == 0 or m == 0 or not goals:
        return []

    # Oriented values: higher is always better.
    raw = [[float(r[c]) for c, _ in goals] for r in rows]
    oriented = [[sign * v for v, (_, sign) in zip(row, goals)] for row in raw]

    g = len(goals)
    lo = [min(o[j] for o in oriented) for j in range(g)]
    hi = [max(o[j] for o in oriented) for j in range(g)]

    # Normalised scores in [0, 1]; a constant goal contributes 0 to everyone.
    scores = []
    for o in oriented:
        s = []
        for j in range(g):
            span = hi[j] - lo[j]
            s.append((o[j] - lo[j]) / span if span > 0 else 0.0)
        scores.append(sum(s) / g)

    # Pareto front: rows not dominated by any other row.
    front = []
    for i in range(n):
        oi = oriented[i]
        dominated = False
        for j in range(n):
            if j == i:
                continue
            oj = oriented[j]
            if all(oj[k] >= oi[k] for k in range(g)) and any(oj[k] > oi[k] for k in range(g)):
                dominated = True
                break
        if not dominated:
            front.append(i)

    front.sort(key=lambda i: (-scores[i], i))
    return front[:m]

