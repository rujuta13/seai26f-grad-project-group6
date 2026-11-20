"""Step 1: flag-family clustering and best-row selection for MOOT data."""

import csv
import math
import random

_K_FAMILIES = 5        # "a handful" of families
_RESTARTS = 5          # k-means restarts; keep the lowest-SSE solution
_ITERS = 100           # max Lloyd iterations per restart
_SEED = 12345          # fixed seed so results are reproducible
_BEST_FRACTION = 0.05  # at most 5% of rows are reported as best


def _cell(text):
    """Return a float if the cell parses as one, otherwise the stripped string."""
    try:
        return float(text)
    except ValueError:
        return text


def _read(path):
    """Read a MOOT csv. Returns (header, rows); rows are dicts {column: value}."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader, [])]
        rows = []
        for raw in reader:
            if not raw:  # blank line, not a data row
                continue
            row = {}
            for name, cell in zip(header, raw):
                row[name] = _cell(cell.strip())
            rows.append(row)
    return header, rows


def _classify(header):
    """Split header into (inputs, max_goals, min_goals). Names ending in X are ignored."""
    inputs, goals_max, goals_min = [], [], []
    for name in header:
        if name.endswith("+"):
            goals_max.append(name)
        elif name.endswith("-"):
            goals_min.append(name)
        elif name.endswith("X"):
            continue
        else:
            inputs.append(name)
    return inputs, goals_max, goals_min


def _features(rows, inputs):
    """Build one feature tuple per row from the input columns.

    Numeric columns are min-max scaled to [0, 1] (constant columns become 0).
    Non-numeric columns are one-hot encoded over their distinct values.
    """
    feats = [[] for _ in rows]
    for name in inputs:
        values = [r.get(name, "") for r in rows]
        if all(isinstance(v, float) for v in values):
            lo, hi = min(values), max(values)
            span = hi - lo
            for i, v in enumerate(values):
                feats[i].append((v - lo) / span if span > 0 else 0.0)
        else:
            strs = [str(v) for v in values]
            cats = sorted(set(strs))
            for i, s in enumerate(strs):
                feats[i].extend(1.0 if s == c else 0.0 for c in cats)
    return [tuple(f) for f in feats]


def _dist2(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _kmeans_pp_init(points, k, rng):
    """k-means++ seeding. Returns k centers as lists."""
    first = points[rng.randrange(len(points))]
    centers = [list(first)]
    d2 = [_dist2(p, centers[0]) for p in points]
    while len(centers) < k:
        total = sum(d2)
        if total <= 0:
            idx = rng.randrange(len(points))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(points) - 1
            for i, d in enumerate(d2):
                acc += d
                if acc >= r:
                    idx = i
                    break
        centers.append(list(points[idx]))
        d2 = [min(d, _dist2(p, centers[-1])) for d, p in zip(d2, points)]
    return centers


def _lloyd(points, centers, iters):
    """Lloyd's algorithm. Returns (labels, sum of squared errors)."""
    n = len(points)
    k = len(centers)
    dim = len(points[0]) if points else 0
    labels = [-1] * n
    for _ in range(iters):
        changed = False
        for i, p in enumerate(points):
            best_c, best_d = 0, math.inf
            for c in range(k):
                d = _dist2(p, centers[c])
                if d < best_d:
                    best_d, best_c = d, c
            if labels[i] != best_c:
                labels[i] = best_c
                changed = True
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for lab, p in zip(labels, points):
            counts[lab] += 1
            s = sums[lab]
            for j in range(dim):
                s[j] += p[j]
        for c in range(k):
            if counts[c]:
                centers[c] = [v / counts[c] for v in sums[c]]
        if not changed:
            break
    sse = sum(_dist2(p, centers[lab]) for p, lab in zip(points, labels))
    return labels, sse


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Clusters rows by their input (flag) columns using k-means with k-means++
    seeding and several restarts; the lowest-SSE solution is kept.
    """
    header, rows = _read(train_csv)
    n = len(rows)
    if n == 0:
        return []
    inputs, _, _ = _classify(header)
    points = _features(rows, inputs)
    distinct = len(set(points))
    k = max(1, min(_K_FAMILIES, n, distinct))
    rng = random.Random(_SEED)
    best_labels, best_sse = None, math.inf
    for _ in range(_RESTARTS):
        centers = _kmeans_pp_init(points, k, rng)
        labels, sse = _lloyd(points, centers, _ITERS)
        if best_labels is None or sse < best_sse:
            best_labels, best_sse = labels, sse
    return best_labels


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows, best on all goals together.

    Each goal is min-max normalised so that higher is better (minimised goals
    are negated). Rows are taken from successive Pareto fronts (non-dominated
    layers), best layer first, ordered within a layer by total normalised score.
    At most 5% of rows are returned, best first.
    """
    header, rows = _read(train_csv)
    n = len(rows)
    cap = int(_BEST_FRACTION * n)
    if cap <= 0:
        return []
    _, goals_max, goals_min = _classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not goals:
        return []

    cols = []
    for name, sign in goals:
        vals = []
        for r in rows:
            v = r.get(name)
            if isinstance(v, float) and not math.isnan(v):
                vals.append(v * sign)
            else:
                vals.append(None)
        present = [v for v in vals if v is not None]
        col = []
        if not present:
            col = [0.0] * n
        else:
            lo, hi = min(present), max(present)
            span = hi - lo
            for v in vals:
                if v is None:
                    col.append(0.0)  # missing counts as worst
                elif span > 0:
                    col.append((v - lo) / span)
                else:
                    col.append(1.0)
        cols.append(col)
    scores = [[col[i] for col in cols] for i in range(n)]
    totals = [sum(s) for s in scores]

    def dominates(a, b):
        strictly = False
        for x, y in zip(a, b):
            if x < y:
                return False
            if x > y:
                strictly = True
        return strictly

    order = []
    remaining = list(range(n))
    while remaining and len(order) < cap:
        front = [
            i for i in remaining
            if not any(dominates(scores[j], scores[i]) for j in remaining if j != i)
        ]
        front.sort(key=lambda i: (-totals[i], i))
        order.extend(front)
        front_set = set(front)
        remaining = [i for i in remaining if i not in front_set]
    return order[:cap]

