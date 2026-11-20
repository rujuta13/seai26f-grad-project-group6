"""step1.py: families (k-means clusters over the input flags) and best
(Pareto-ranked rows over the goal columns).

This file is self-contained. The earlier steps' module names and helpers are
not visible in this message, so nothing is imported from them.
"""
from __future__ import annotations

import csv
import math
import random

K_FAMILIES = 5          # "a handful of families"
BEST_PERCENT = 5        # at most 5% of rows are reported as best
_SEED = 1               # fixed seed so results are reproducible


def _num(v) -> float | None:
    """Return v as a finite float, or None if it is not a usable number."""
    if isinstance(v, float) and math.isfinite(v):
        return v
    return None


def _read(path: str) -> tuple[list[str], list[dict]]:
    """Read a MOOT csv. Numeric cells become floats; other cells stay strings."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        rows = []
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if k is None:
                    continue
                key = k.strip()
                text = (v or "").strip()
                try:
                    row[key] = float(text)
                except ValueError:
                    row[key] = text
            rows.append(row)
    return header, rows


def _columns(header: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split columns into (inputs, goals to maximise, goals to minimise).

    Names ending in "X" are ignored. Names ending in "+" or "-" are goals.
    Every other column is an input.
    """
    inputs, gmax, gmin = [], [], []
    for c in header:
        if c.endswith("X"):
            continue
        if c.endswith("+"):
            gmax.append(c)
        elif c.endswith("-"):
            gmin.append(c)
        else:
            inputs.append(c)
    return inputs, gmax, gmin


def _features(rows: list[dict], cols: list[str]) -> list[list[float]]:
    """Scale each input column to [0, 1]. Strings become category codes first."""
    if not rows or not cols:
        return [[] for _ in rows]
    scaled = []
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        if all(isinstance(v, float) for v in vals):
            nums = list(vals)
        else:
            cats = {s: i for i, s in enumerate(sorted({str(v) for v in vals}))}
            nums = [float(cats[str(v)]) for v in vals]
        lo, hi = min(nums), max(nums)
        span = hi - lo
        scaled.append([(x - lo) / span if span > 0 else 0.0 for x in nums])
    return [[col[i] for col in scaled] for i in range(len(rows))]


def _dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _kmeans(points: list[list[float]], k: int, seed: int = _SEED,
            iters: int = 100) -> list[int]:
    """Plain k-means with k-means++ seeding. Returns one label per point."""
    n = len(points)
    if n == 0:
        return []
    k = max(1, min(k, n))
    dim = len(points[0])
    rng = random.Random(seed)

    # k-means++ seeding
    centers = [list(points[rng.randrange(n)])]
    d2 = [_dist2(p, centers[0]) for p in points]
    while len(centers) < k:
        total = sum(d2)
        if total <= 0:
            idx = rng.randrange(n)
        else:
            r = rng.random() * total
            acc = 0.0
            idx = n - 1
            for i, d in enumerate(d2):
                acc += d
                if acc >= r:
                    idx = i
                    break
        centers.append(list(points[idx]))
        d2 = [min(d2[i], _dist2(points[i], centers[-1])) for i in range(n)]

    labels = [-1] * n
    for _ in range(iters):
        new = [min(range(k), key=lambda c: _dist2(p, centers[c])) for p in points]
        if new == labels:
            break
        labels = new
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for p, c in zip(points, labels):
            counts[c] += 1
            for j in range(dim):
                sums[c][j] += p[j]
        for c in range(k):
            if counts[c]:
                centers[c] = [s / counts[c] for s in sums[c]]
            else:
                # Re-seed an empty cluster at the point farthest from its center.
                far = max(range(n), key=lambda i: _dist2(points[i], centers[labels[i]]))
                centers[c] = list(points[far])
    return labels


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    header, rows = _read(str(train_csv))
    inputs, _, _ = _columns(header)
    return _kmeans(_features(rows, inputs), K_FAMILIES)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows, at most 5% of rows.

    Rows are ranked by Pareto fronts over all goals. Front 0 is the set of rows
    no other row dominates on every goal at once. Front 1 is the set left when
    front 0 is removed, and so on. Within a front, rows are ordered by their
    mean min-max-normalised goal score (higher is better). The result is
    returned best first.
    """
    header, rows = _read(str(train_csv))
    _, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    limit = (n * BEST_PERCENT) // 100
    if n == 0 or not goals or limit == 0:
        return []

    # Signed objective vectors: larger is better on every axis.
    # A missing or non-numeric goal value counts as the worst possible.
    obj = []
    for r in rows:
        vec = []
        for c in gmax:
            v = _num(r.get(c))
            vec.append(v if v is not None else -math.inf)
        for c in gmin:
            v = _num(r.get(c))
            vec.append(-v if v is not None else -math.inf)
        obj.append(vec)

    # Mean normalised score in [0, 1] per row, used to order within a front.
    m = len(goals)
    norm = [[0.0] * m for _ in range(n)]
    for j in range(m):
        col = [o[j] for o in obj]
        fin = [x for x in col if math.isfinite(x)]
        if not fin:
            continue
        lo, hi = min(fin), max(fin)
        span = hi - lo
        for i, x in enumerate(col):
            if math.isfinite(x):
                norm[i][j] = (x - lo) / span if span > 0 else 1.0
    score = [sum(v) / m for v in norm]

    def dominates(a: list[float], b: list[float]) -> bool:
        strictly = False
        for x, y in zip(a, b):
            if x < y:
                return False
            if x > y:
                strictly = True
        return strictly

    chosen: list[int] = []
    remaining = list(range(n))
    while remaining and len(chosen) < limit:
        front = [
            i for i in remaining
            if not any(dominates(obj[j], obj[i]) for j in remaining if j != i)
        ]
        front.sort(key=lambda i: (-score[i], i))
        room = limit - len(chosen)
        chosen.extend(front[:room])
        in_front = set(front)
        remaining = [i for i in remaining if i not in in_front]
    return chosen

