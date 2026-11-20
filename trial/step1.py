"""step1.py: families (clusters) and unusually good rows of a MOOT csv.

Standard library only (Python 3.13). No earlier-step module was supplied
with this request, so the small csv and MOOT helpers are defined here.
"""
from __future__ import annotations

import csv
import math
import random

_SEED = 1
_MAX_K = 8
_RESTARTS = 3
_SIL_SAMPLE = 250
_BEST_NUM, _BEST_DEN = 5, 100  # "at most 5% of rows"


def _read(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [r for r in reader if any(c.strip() for c in r)]
    return header, rows


def _classify(header: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    """Return (input column indices, [(goal column index, sign)])."""
    inputs: list[int] = []
    goals: list[tuple[int, int]] = []
    for j, name in enumerate(header):
        if name.endswith("+"):
            goals.append((j, 1))
        elif name.endswith("-"):
            goals.append((j, -1))
        elif name.endswith("X"):
            continue
        else:
            inputs.append(j)
    return inputs, goals


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _encode_inputs(rows: list[list[str]], cols: list[int]) -> list[list[float]]:
    codes: dict[int, dict[str, int]] = {}
    X: list[list[float]] = []
    for r in rows:
        vec: list[float] = []
        for j in cols:
            v = _to_float(r[j])
            if v is None:
                table = codes.setdefault(j, {})
                v = float(table.setdefault(r[j], len(table)))
            vec.append(v)
        X.append(vec)
    return X


def _minmax(X: list[list[float]]) -> list[list[float]]:
    if not X:
        return X
    m = len(X[0])
    lo = [min(r[k] for r in X) for k in range(m)]
    hi = [max(r[k] for r in X) for k in range(m)]
    out: list[list[float]] = []
    for r in X:
        out.append([
            (r[k] - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0.0
            for k in range(m)
        ])
    return out


def _goal_scores(rows: list[list[str]],
                 goals: list[tuple[int, int]]) -> list[list[float]]:
    """Normalise each goal to [0, 1] where 1 is best; missing values score 0."""
    n = len(rows)
    columns: list[list[float]] = []
    for j, sign in goals:
        vals = [_to_float(r[j]) for r in rows]
        ok = [v for v in vals if v is not None and not math.isnan(v)]
        col = [0.0] * n
        if ok:
            lo, hi = min(ok), max(ok)
            span = hi - lo
            for i, v in enumerate(vals):
                if v is None or math.isnan(v):
                    continue
                if span == 0:
                    col[i] = 1.0
                elif sign > 0:
                    col[i] = (v - lo) / span
                else:
                    col[i] = (hi - v) / span
        columns.append(col)
    return [[col[i] for col in columns] for i in range(n)]


def _kmeans(X: list[list[float]], k: int, rng: random.Random,
            iters: int = 100) -> tuple[list[int], float]:
    n, dims = len(X), len(X[0])

    def sqdist(a: list[float], b: list[float]) -> float:
        return sum((x - y) * (x - y) for x, y in zip(a, b))

    # k-means++ initialisation
    centers = [list(X[rng.randrange(n)])]
    d2 = [sqdist(x, centers[0]) for x in X]
    while len(centers) < k:
        total = sum(d2)
        if total > 0:
            t = rng.uniform(0.0, total)
            acc, pick = 0.0, n - 1
            for i, d in enumerate(d2):
                acc += d
                if acc >= t:
                    pick = i
                    break
        else:
            pick = rng.randrange(n)
        centers.append(list(X[pick]))
        d2 = [min(d2[i], sqdist(X[i], centers[-1])) for i in range(n)]

    labels = [-1] * n
    for it in range(iters):
        changed = False
        for i, x in enumerate(X):
            best_c, best_d = 0, math.inf
            for c in range(k):
                d = sqdist(x, centers[c])
                if d < best_d:
                    best_c, best_d = c, d
            if best_c != labels[i]:
                labels[i] = best_c
                changed = True
        if not changed and it > 0:
            break
        sums = [[0.0] * dims for _ in range(k)]
        counts = [0] * k
        for x, c in zip(X, labels):
            counts[c] += 1
            s = sums[c]
            for d in range(dims):
                s[d] += x[d]
        for c in range(k):
            if counts[c]:
                centers[c] = [v / counts[c] for v in sums[c]]
            else:
                centers[c] = list(X[rng.randrange(n)])

    inertia = sum(sqdist(x, centers[c]) for x, c in zip(X, labels))
    return labels, inertia


def _best_kmeans(X: list[list[float]], k: int,
                 rng: random.Random) -> list[int]:
    best_labels, best_inertia = None, math.inf
    for _ in range(_RESTARTS):
        labels, inertia = _kmeans(X, k, rng)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    assert best_labels is not None
    return best_labels


def _silhouette(X: list[list[float]], labels: list[int],
                rng: random.Random) -> float:
    n = len(X)
    idx = list(range(n))
    if n > _SIL_SAMPLE:
        idx = sorted(rng.sample(idx, _SIL_SAMPLE))
    pts = [X[i] for i in idx]
    lab = [labels[i] for i in idx]
    k = max(lab) + 1
    members: list[list[int]] = [[] for _ in range(k)]
    for i, c in enumerate(lab):
        members[c].append(i)
    if sum(1 for m in members if m) < 2:
        return -1.0

    total = 0.0
    for i, p in enumerate(pts):
        own = members[lab[i]]
        if len(own) <= 1:
            continue
        a = sum(math.dist(p, pts[j]) for j in own if j != i) / (len(own) - 1)
        b = math.inf
        for c, m in enumerate(members):
            if c == lab[i] or not m:
                continue
            mean = sum(math.dist(p, pts[j]) for j in m) / len(m)
            if mean < b:
                b = mean
        top = max(a, b)
        if top > 0:
            total += (b - a) / top
    return total / len(pts)


def _relabel(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for c in labels:
        if c not in mapping:
            mapping[c] = len(mapping)
        out.append(mapping[c])
    return out


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Clusters the input columns with k-means, choosing k in 2..8 by
    silhouette score. Labels are contiguous integers starting at 0.
    """
    header, rows = _read(train_csv)
    inputs, _ = _classify(header)
    n = len(rows)
    if n == 0:
        return []
    if not inputs:
        return [0] * n

    X = _minmax(_encode_inputs(rows, inputs))
    rng = random.Random(_SEED)
    chosen: list[int] | None = None
    chosen_score = -math.inf
    for k in range(2, min(_MAX_K, n) + 1):
        labels = _best_kmeans(X, k, rng)
        score = _silhouette(X, labels, rng)
        if score > chosen_score:
            chosen, chosen_score = labels, score
    if chosen is None:
        return [0] * n
    return _relabel(chosen)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together).

    Uses the Pareto front over the normalised goals (maximise and minimise
    goals both scaled so that 1 is best). If the front has more rows than
    5% of the data, it keeps those with the highest mean normalised score.
    """
    header, rows = _read(train_csv)
    _, goals = _classify(header)
    n = len(rows)
    cap = n * _BEST_NUM // _BEST_DEN
    if n == 0 or not goals or cap == 0:
        return []

    S = _goal_scores(rows, goals)
    front: list[int] = []
    for i in range(n):
        si = S[i]
        dominated = False
        for j in range(n):
            if j == i:
                continue
            sj = S[j]
            if all(b >= a for a, b in zip(si, sj)) and any(
                    b > a for a, b in zip(si, sj)):
                dominated = True
                break
        if not dominated:
            front.append(i)

    front.sort(key=lambda i: (-sum(S[i]), i))
    return front[:cap]

