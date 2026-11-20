"""step1.py: family labels and best-row selection for the MOOT flag data.

The helpers from earlier steps are not available in this context, so the
small CSV, clustering and dominance utilities needed here are defined inline.
Standard library only.
"""
import csv
import random

K_FAMILIES = 5        # "a handful" of families
BEST_FRACTION = 0.05  # best() returns at most 5% of rows


def _load(path):
    """Read a MOOT csv. Numeric cells become floats; others stay strings."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = []
        for raw in reader:
            row = {}
            for c in cols:
                v = raw[c]
                try:
                    row[c] = float(v)
                except (TypeError, ValueError):
                    row[c] = v
            rows.append(row)
    return cols, rows


def _columns(cols, rows):
    """Split columns into numeric inputs, maximise goals and minimise goals.

    Columns ending in "X" are ignored. Columns ending in "+" are maximised,
    columns ending in "-" are minimised, and every other column is an input.
    Only inputs whose values are all numeric are returned.
    """
    inputs, maxi, mini = [], [], []
    for c in cols:
        if c.endswith("X"):
            continue
        if c.endswith("+"):
            maxi.append(c)
        elif c.endswith("-"):
            mini.append(c)
        else:
            inputs.append(c)
    numeric_inputs = [c for c in inputs
                      if all(isinstance(r[c], float) for r in rows)]
    return numeric_inputs, maxi, mini


def _sqd(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _kmeanspp(X, k, rng):
    """k-means++ seeding."""
    centers = [list(X[rng.randrange(len(X))])]
    while len(centers) < k:
        dist = [min(_sqd(x, c) for c in centers) for x in X]
        total = sum(dist)
        if total <= 0:
            centers.append(list(X[rng.randrange(len(X))]))
            continue
        r = rng.uniform(0, total)
        acc = 0.0
        pick = len(X) - 1
        for i, dx in enumerate(dist):
            acc += dx
            if acc >= r:
                pick = i
                break
        centers.append(list(X[pick]))
    return centers


def _centroids(X, labels, k, old):
    d = len(X[0])
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    for x, l in zip(X, labels):
        counts[l] += 1
        s = sums[l]
        for t, v in enumerate(x):
            s[t] += v
    return [[v / counts[j] for v in sums[j]] if counts[j] else old[j]
            for j in range(k)]


def _kmeans(X, k, restarts=5, iters=100, seed=0):
    """Plain k-means with several restarts. Returns labels in 0..k-1."""
    rng = random.Random(seed)
    best_labels, best_inertia = None, None
    for _ in range(restarts):
        centers = _kmeanspp(X, k, rng)
        labels = None
        for _ in range(iters):
            new = [min(range(k), key=lambda j: _sqd(x, centers[j])) for x in X]
            if new == labels:
                break
            labels = new
            centers = _centroids(X, labels, k, centers)
        inertia = sum(_sqd(x, centers[l]) for x, l in zip(X, labels))
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    cols, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    inputs, _, _ = _columns(cols, rows)
    if not inputs:
        return [0] * n
    X = [[r[c] for c in inputs] for r in rows]
    k = min(K_FAMILIES, n)
    return [int(l) for l in _kmeans(X, k)]


def _dominates(a, b):
    """a dominates b when a is at least as good everywhere and better somewhere.
    Both vectors are oriented so that higher is better."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows.

    "Best on all goals together" is taken as the non-dominated (Pareto) front.
    If the front is larger than the cap, the front rows are ranked by their
    normalised goal sum and the top ones are kept.
    """
    cols, rows = _load(train_csv)
    n = len(rows)
    cap = int(BEST_FRACTION * n)
    if n == 0 or cap == 0:
        return []
    _, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals:
        return []

    # Orient every goal so that higher is better.
    V = [[sgn * float(r[c]) for c, sgn in goals] for r in rows]

    front = [i for i in range(n)
             if not any(_dominates(V[j], V[i]) for j in range(n) if j != i)]

    if len(front) > cap:
        g = len(goals)
        lo = [min(V[i][t] for i in range(n)) for t in range(g)]
        hi = [max(V[i][t] for i in range(n)) for t in range(g)]

        def score(i):
            s = 0.0
            for t in range(g):
                span = hi[t] - lo[t]
                s += (V[i][t] - lo[t]) / span if span > 0 else 1.0
            return s

        front.sort(key=lambda i: (-score(i), i))
        front = front[:cap]

    return sorted(front)

