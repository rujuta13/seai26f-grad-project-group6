"""Step 11: the client's two questions, answered on large player files.

families(train_csv): sorts rows into types of player with k-means (K families).
  Numeric input columns are min-max scaled to [0, 1] so no column dominates the
  distances. Text input columns are one-hot encoded when they have at most
  MAX_LEVELS distinct values. Text columns with more values (identifier-like,
  e.g. a url or a name) are skipped. Constant columns are dropped.

best(train_csv): the "strong and good from the spot" players. These are rows on
  the Pareto front of the goals (no other row is at least as good on every goal
  and better on one), capped at 5% of rows and ordered by mean normalised goal
  score. The front is found by sorting on the goal sum, so each row is checked
  only against the front found so far. This keeps 17,000 rows fast.

Rows with a missing or non-numeric value in an input or goal cell are handled
without crashing: goal cells that are unusable exclude the row from best().
"""
import csv
import math
import random

K = 5                  # number of families reported
SEED = 1               # fixed seed so results are reproducible
MAX_ITER = 200
MAX_LEVELS = 25        # text columns with more distinct values are skipped (identifier-like)
FRONT_CAP = 0.05       # at most this share of rows is reported as "good"


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


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


def _features(rows, inputs):
    """Encode input cells as rows of floats in [0, 1], one column per feature.
    Constant features are dropped. Returns [] if nothing is left."""
    n = len(rows)
    cols = []
    for c in inputs:
        raw = [r.get(c) for r in rows]
        nums = [_finite(v) for v in raw]
        if all(x is not None for x in nums):
            cols.append(nums)
            continue
        texts = [_text(v) for v in raw]
        levels = sorted(set(texts))
        if len(levels) > MAX_LEVELS:
            continue  # identifier-like text: one-hot would explode and carry no type
        index = {lv: j for j, lv in enumerate(levels)}
        block = [[0.0] * n for _ in levels]
        for i, t in enumerate(texts):
            block[index[t]][i] = 1.0
        cols.extend(block)

    kept = []
    for col in cols:
        lo, hi = min(col), max(col)
        if hi > lo:
            span = hi - lo
            kept.append([(x - lo) / span for x in col])
    if not kept:
        return []
    return [list(t) for t in zip(*kept)]


def _kmeans(X, k, seed):
    """Plain k-means with k-means++ initialisation, standard library only."""
    n = len(X)
    rng = random.Random(seed)

    # k-means++ seeding, keeping each point's squared distance to its nearest centre.
    centers = [list(X[rng.randrange(n)])]
    near = [math.dist(x, centers[0]) ** 2 for x in X]
    while len(centers) < k:
        total = math.fsum(near)
        if total > 0:
            r = rng.random() * total
            acc = 0.0
            pick = n - 1
            for i, dv in enumerate(near):
                acc += dv
                if acc >= r:
                    pick = i
                    break
        else:
            pick = rng.randrange(n)
        c = list(X[pick])
        centers.append(c)
        for i, x in enumerate(X):
            dv = math.dist(x, c) ** 2
            if dv < near[i]:
                near[i] = dv

    labels = [-1] * n
    for _ in range(MAX_ITER):
        changed = False
        for i, x in enumerate(X):
            best_c = min(range(k), key=lambda j: math.dist(x, centers[j]))
            if best_c != labels[i]:
                labels[i] = best_c
                changed = True
        if not changed:
            break
        groups = [[] for _ in range(k)]
        for x, c in zip(X, labels):
            groups[c].append(x)
        for c in range(k):
            if groups[c]:
                m = len(groups[c])
                centers[c] = [sum(col) / m for col in zip(*groups[c])]
            else:
                centers[c] = list(X[rng.randrange(n)])
    return labels


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    rows, inputs, _ = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    X = _features(rows, inputs)
    if not X or not X[0]:
        return [0] * n
    distinct = len({tuple(x) for x in X})
    k = max(1, min(K, distinct))
    if k == 1:
        return [0] * n
    return _kmeans(X, k, SEED)


def _dominates(q, p):
    """True if q is at least as good as p on every goal and better on at least one."""
    return all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p))


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows. Pareto front over the goals, ranked by mean normalised
    goal score (ties by row index)."""
    rows, _, goals = _load(train_csv)
    n = len(rows)
    m = int(FRONT_CAP * n)
    if n == 0 or m == 0 or not goals:
        return []

    # Oriented goal vectors (higher is always better); rows with unusable goals are left out.
    oriented = {}
    for i, r in enumerate(rows):
        vec = []
        for c, sign in goals:
            x = _finite(r.get(c))
            if x is None:
                break
            vec.append(sign * x)
        else:
            oriented[i] = vec
    if not oriented:
        return []

    g = len(goals)
    lo = [min(v[j] for v in oriented.values()) for j in range(g)]
    hi = [max(v[j] for v in oriented.values()) for j in range(g)]

    scores = {}
    for i, v in oriented.items():
        total = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            if span > 0:
                total += (v[j] - lo[j]) / span
        scores[i] = total / g

    # A row can only be dominated by a row with a strictly larger goal sum, so
    # visiting rows in decreasing sum order and checking against the front found
    # so far gives the exact Pareto front (dominance is transitive).
    order = sorted(oriented, key=lambda i: -sum(oriented[i]))
    front = []
    for i in order:
        p = oriented[i]
        if any(_dominates(oriented[j], p) for j in front):
            continue
        front.append(i)

    front.sort(key=lambda i: (-scores[i], i))
    return front[:m]

