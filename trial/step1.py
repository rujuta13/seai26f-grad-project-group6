"""Step 1: flag-family clustering and best-row selection.

Standard library only. The earlier-step modules were not provided, so this
file is self-contained and does not import from them.
"""

import csv
import random


K_FAMILIES = 5           # number of flag families reported
SEED = 1                 # fixed seed so results are reproducible
KMEANS_ITERS = 100


def _load(train_csv):
    """Return (header, rows). Numeric cells become floats; others stay strings."""
    with open(train_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for cells in reader:
            if not cells or all(c.strip() == "" for c in cells):
                continue
            row = {}
            for name, cell in zip(header, cells):
                try:
                    row[name] = float(cell)
                except ValueError:
                    row[name] = cell
            rows.append(row)
    return header, rows


def _columns(header):
    """Split columns into (inputs, goals). Goals are (name, sign): +1 max, -1 min."""
    inputs, goals = [], []
    for name in header:
        if name.endswith("X"):
            continue
        if name.endswith("+"):
            goals.append((name, 1.0))
        elif name.endswith("-"):
            goals.append((name, -1.0))
        else:
            inputs.append(name)
    return inputs, goals


def _input_matrix(rows, inputs):
    """One numeric vector per row. Non-numeric cells are coded by first appearance."""
    codes = {}
    matrix = []
    for row in rows:
        vec = []
        for c in inputs:
            v = row[c]
            if isinstance(v, float):
                vec.append(v)
            else:
                table = codes.setdefault(c, {})
                vec.append(float(table.setdefault(v, len(table))))
        matrix.append(vec)
    return matrix


def _sqdist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _kmeans(points, k, rng, iters=KMEANS_ITERS):
    """Lloyd's algorithm with k-means++ seeding. Returns one label per point."""
    n = len(points)
    k = min(k, n)
    dim = len(points[0])

    # k-means++ seeding
    centers = [list(points[rng.randrange(n)])]
    while len(centers) < k:
        d = [min(_sqdist(p, c) for c in centers) for p in points]
        total = sum(d)
        if total == 0:
            centers.append(list(points[rng.randrange(n)]))
            continue
        r = rng.uniform(0, total)
        acc = 0.0
        pick = n - 1
        for i, di in enumerate(d):
            acc += di
            if acc >= r:
                pick = i
                break
        centers.append(list(points[pick]))

    labels = [-1] * n
    for _ in range(iters):
        changed = False
        for i, p in enumerate(points):
            best_c = min(range(k), key=lambda c: _sqdist(p, centers[c]))
            if best_c != labels[i]:
                labels[i] = best_c
                changed = True
        if not changed:
            break

        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for p, lab in zip(points, labels):
            counts[lab] += 1
            s = sums[lab]
            for j, x in enumerate(p):
                s[j] += x
        for c in range(k):
            if counts[c]:
                centers[c] = [v / counts[c] for v in sums[c]]
            else:
                # Reseed an empty cluster from a random point.
                centers[c] = list(points[rng.randrange(n)])
    return labels


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    header, rows = _load(train_csv)
    if not rows:
        return []
    inputs, _ = _columns(header)
    points = _input_matrix(rows, inputs)
    if not inputs:
        return [0] * len(rows)
    rng = random.Random(SEED)
    return _kmeans(points, K_FAMILIES, rng)


def _normalised_goals(rows, goals):
    """Per row, a vector of goal values in [0, 1] where higher is always better."""
    cols = []
    for name, sign in goals:
        vals = [sign * float(r[name]) for r in rows]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        cols.append([(v - lo) / span if span > 0 else 0.0 for v in vals])
    return [list(t) for t in zip(*cols)]


def _dominates(a, b):
    """True if a is at least as good as b on every goal and strictly better on one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _pareto_layers(points):
    """Non-dominated sorting. Returns the layer index of each point (0 = Pareto front)."""
    n = len(points)
    layer_of = [0] * n
    remaining = list(range(n))
    layer = 0
    while remaining:
        front = []
        for i in remaining:
            pi = points[i]
            if not any(_dominates(points[j], pi) for j in remaining if j != i):
                front.append(i)
        for i in front:
            layer_of[i] = layer
        front_set = set(front)
        remaining = [i for i in remaining if i not in front_set]
        layer += 1
    return layer_of


def best(train_csv) -> list[int]:
    """Row indices of rows that are best on all goals together (Pareto front),
    ranked by mean normalised goal value and capped at 5% of rows."""
    header, rows = _load(train_csv)
    n = len(rows)
    cap = n // 20  # floor(5% of n), computed without float error
    _, goals = _columns(header)
    if n == 0 or cap == 0 or not goals:
        return []

    points = _normalised_goals(rows, goals)
    layer_of = _pareto_layers(points)
    score = [sum(p) / len(p) for p in points]

    front = [i for i in range(n) if layer_of[i] == 0]
    front.sort(key=lambda i: (-score[i], i))
    return front[:cap]

