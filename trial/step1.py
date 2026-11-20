import csv
import random


def _load(train_csv):
    """Return (header, rows) where rows are dicts {column: float or str}."""
    with open(train_csv, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = []
        for raw in reader:
            if not raw or all(not c.strip() for c in raw):
                continue
            row = {}
            for name, cell in zip(header, raw):
                cell = cell.strip()
                try:
                    row[name] = float(cell)
                except ValueError:
                    row[name] = cell
            rows.append(row)
    return header, rows


def _as_float(v):
    return v if isinstance(v, float) else 0.0


def _sq_dist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _kmeans(X, k, seed=1, max_iter=100):
    """Plain Lloyd's k-means with k-means++ seeding. Deterministic for a given seed."""
    rng = random.Random(seed)
    n = len(X)
    centers = [list(X[rng.randrange(n)])]
    while len(centers) < k:
        d = [min(_sq_dist(x, c) for c in centers) for x in X]
        total = sum(d)
        if total <= 0:
            centers.append(list(X[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        pick = n - 1
        for i, di in enumerate(d):
            acc += di
            if acc >= r:
                pick = i
                break
        centers.append(list(X[pick]))

    labels = [-1] * n
    for _ in range(max_iter):
        changed = False
        for i, x in enumerate(X):
            best_c = min(range(k), key=lambda c: _sq_dist(x, centers[c]))
            if labels[i] != best_c:
                labels[i] = best_c
                changed = True
        if not changed:
            break
        for c in range(k):
            members = [X[i] for i in range(n) if labels[i] == c]
            if members:
                centers[c] = [sum(col) / len(members) for col in zip(*members)]
    return labels


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Clusters are computed on the input (flag) columns only. Labels are
    renumbered 0..k-1 in order of first appearance in the file.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []

    input_cols = [
        c for c in header
        if not (c.endswith("+") or c.endswith("-") or c.endswith("X"))
    ]
    if not input_cols:
        return [0] * n

    X = [[_as_float(r.get(c, 0.0)) for c in input_cols] for r in rows]

    k = min(5, n)
    raw = _kmeans(X, k)

    relabel = {}
    out = []
    for lab in raw:
        if lab not in relabel:
            relabel[lab] = len(relabel)
        out.append(relabel[lab])
    return out


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together).

    Goals ending in '+' are maximised, goals ending in '-' are minimised.
    Rows are ordered by non-dominated front (Pareto layers), then by a
    normalised sum of goal scores within each layer. At most 5% of rows
    are returned.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    cap = int(0.05 * n)
    if cap == 0:
        return []

    goals = []
    for c in header:
        if c.endswith("+"):
            goals.append((c, 1.0))
        elif c.endswith("-"):
            goals.append((c, -1.0))
    if not goals:
        return []

    # Oriented values: higher is always better.
    V = [[sg * _as_float(r.get(c, 0.0)) for c, sg in goals] for r in rows]

    g = len(goals)
    lo = [min(V[i][j] for i in range(n)) for j in range(g)]
    hi = [max(V[i][j] for i in range(n)) for j in range(g)]
    score = []
    for i in range(n):
        s = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            s += 1.0 if span == 0 else (V[i][j] - lo[j]) / span
        score.append(s)

    def dominates(a, b):
        ge = True
        gt = False
        for x, y in zip(a, b):
            if x < y:
                ge = False
                break
            if x > y:
                gt = True
        return ge and gt

    remaining = set(range(n))
    order = []
    while remaining and len(order) < cap:
        front = [
            i for i in remaining
            if not any(dominates(V[j], V[i]) for j in remaining if j != i)
        ]
        if not front:  # cannot happen for a finite partial order; safety
            front = list(remaining)
        front.sort(key=lambda i: (-score[i], i))
        order.extend(front)
        remaining.difference_update(front)

    return order[:cap]

