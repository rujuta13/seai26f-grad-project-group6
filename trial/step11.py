"""Step 11: remember. Player families and best-player rows for the client.

families(): flag-family clustering like step1.families, adjusted for about
17,000 rows:
  - categorical columns with many distinct values (names, URLs, teams) are left
    out, since one-hot encoding them would make nearly every row a unique point;
  - k-means is fitted on a deterministic sample, then every row is assigned to
    its nearest centre.

best(): the same Pareto-layer ranking as step1.best, computed faster. Rows are
visited in descending lexicographic order of their normalised goal scores, so a
dominator is always visited before the rows it dominates. Each row only has to
be checked against the rows already accepted into the current layer.

Results are remembered per file (keyed by path, size and modification time),
so repeated client requests return the stored answer instead of recomputing it.
"""

import math
import os
import random

import step1

_K_FAMILIES = 5        # "a handful" of families
_RESTARTS = 5          # k-means restarts; keep the lowest-SSE solution
_ITERS = 100           # max Lloyd iterations per restart
_SAMPLE = 2000         # rows used to fit the centres (all rows are then labelled)
_MAX_CATS = 40         # text columns with more distinct values are not used as flags
_SEED = 12345          # fixed seed so results are reproducible
_BEST_FRACTION = 0.05  # at most 5% of rows are reported as best

_memory = {}           # (kind, abs path) -> (file stamp, result)


def _remember(kind, train_csv, compute):
    """Return compute(train_csv), reusing the stored result while the file is unchanged."""
    path = os.path.abspath(train_csv)
    st = os.stat(path)
    stamp = (st.st_size, st.st_mtime_ns)
    key = (kind, path)
    hit = _memory.get(key)
    if hit is None or hit[0] != stamp:
        hit = (stamp, compute(train_csv))
        _memory[key] = hit
    return list(hit[1])


# ---------------------------------------------------------------- families

def _usable_inputs(rows, inputs):
    """Keep numeric inputs and text inputs with a manageable number of values."""
    keep = []
    for name in inputs:
        values = [r.get(name) for r in rows]
        if all(isinstance(v, float) for v in values):
            keep.append(name)
        elif len({str(v) for v in values}) <= _MAX_CATS:
            keep.append(name)
    return keep


def _assign(points, centers):
    """Index of the nearest centre for every point, in order."""
    labels = []
    for p in points:
        best_c, best_d = 0, math.inf
        for c, center in enumerate(centers):
            d = step1._dist2(p, center)
            if d < best_d:
                best_d, best_c = d, c
        labels.append(best_c)
    return labels


def _compute_families(train_csv):
    header, rows = step1._read(train_csv)
    n = len(rows)
    if n == 0:
        return []
    inputs, _, _ = step1._classify(header)
    inputs = _usable_inputs(rows, inputs)
    if not inputs:
        return [0] * n

    points = step1._features(rows, inputs)
    rng = random.Random(_SEED)
    if n <= _SAMPLE:
        fit = points
    else:
        fit = [points[i] for i in sorted(rng.sample(range(n), _SAMPLE))]

    k = max(1, min(_K_FAMILIES, n, len(set(fit))))
    best_centers, best_sse = None, math.inf
    for _ in range(_RESTARTS):
        centers = step1._kmeans_pp_init(fit, k, rng)
        _, sse = step1._lloyd(fit, centers, _ITERS)
        if best_centers is None or sse < best_sse:
            best_centers, best_sse = centers, sse
    return _assign(points, best_centers)


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Clusters rows by their input (flag) columns using k-means with k-means++
    seeding and several restarts; the lowest-SSE solution is kept. Text columns
    with very many distinct values (names, URLs, teams) are skipped so they do
    not dominate the distances.
    """
    return _remember("families", train_csv, _compute_families)


# ---------------------------------------------------------------- best

def _dominates(a, b):
    strictly = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strictly = True
    return strictly


def _front(remaining, scores):
    """Rows of `remaining` not dominated by any other row of `remaining`.

    Rows are visited in descending lexicographic order of their score vectors.
    If a dominates b, a's vector is lexicographically larger, so a is visited
    first. Each row is therefore checked only against the rows already accepted.
    """
    ordered = sorted(remaining, key=lambda i: scores[i], reverse=True)
    kept = []
    for i in ordered:
        if not any(_dominates(scores[k], scores[i]) for k in kept):
            kept.append(i)
    return kept


def _compute_best(train_csv):
    header, rows = step1._read(train_csv)
    n = len(rows)
    cap = int(_BEST_FRACTION * n)
    if cap <= 0:
        return []
    _, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not goals:
        return []

    scores = [[0.0] * len(goals) for _ in range(n)]
    for c, (name, sign) in enumerate(goals):
        vals = []
        for r in rows:
            v = r.get(name)
            if isinstance(v, float) and not math.isnan(v):
                vals.append(v * sign)
            else:
                vals.append(None)
        present = [v for v in vals if v is not None]
        if not present:
            continue  # all missing: every row scores 0 on this goal
        lo, hi = min(present), max(present)
        span = hi - lo
        for i, v in enumerate(vals):
            if v is None:
                continue  # missing counts as worst
            scores[i][c] = (v - lo) / span if span > 0 else 1.0
    totals = [sum(s) for s in scores]

    order = []
    remaining = list(range(n))
    while remaining and len(order) < cap:
        front = _front(remaining, scores)
        front.sort(key=lambda i: (-totals[i], i))
        order.extend(front)
        front_set = set(front)
        remaining = [i for i in remaining if i not in front_set]
    return order[:cap]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows, best on all goals together.

    Each goal is min-max normalised so that higher is better (minimised goals
    are negated). Rows are taken from successive Pareto fronts (non-dominated
    layers), best layer first, ordered within a layer by total normalised score.
    At most 5% of rows are returned, best first.
    """
    return _remember("best", train_csv, _compute_best)

