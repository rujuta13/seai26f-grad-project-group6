"""Step 17: choose. Find a good row while calling label() at most `budget` times.

blind_csv has the inputs of every candidate build, with every goal cell set to '?'.
label(i) benchmarks row i and returns its goals as a dict. The budget is spent in two
phases:
  1. Exploration: maximin (farthest-point) sampling over the encoded inputs, so the
     first benchmarks cover the input space.
  2. Adaptive search: each new row maximises (predicted score from the nearest
     benchmarked rows) + KAPPA * (distance to the nearest benchmarked row), with
     KAPPA cycling through exploit, exploit, explore.

Scores use the same scale as step1: each goal is min-max scaled over the benchmarked
rows (0 = worst, 1 = best) and averaged. If `goal` names one goal, only that goal is
optimised. The returned index is the benchmarked row with the highest score.

Input encoding: numeric columns are min-max scaled. Identifier-like numeric columns
(almost one distinct value per row) and text columns with many levels (names, urls,
teams, nations) are skipped, so the distances stay meaningful and fast to compute.
"""
import math
import random

from step1 import _load

SEED = 1              # fixed seed so results are reproducible
INIT_FRAC = 0.5       # share of the budget spent on the exploration phase
K_NEAR = 3            # benchmarked neighbours used to predict an unbenchmarked row's score
KAPPA = (0.0, 0.0, 0.5)  # exploit, exploit, explore cycle for the adaptive phase
EPS = 1e-9            # keeps neighbour weights finite when a distance is zero
MAX_LEVELS = 25       # text columns with more distinct values are skipped
NUMERIC_FRAC = 0.9    # a column is numeric if this share of its cells parse as numbers
ID_FRAC = 0.9         # a numeric column with more distinct values than this share is an id


def _finite(v):
    """The cell as a finite float, or None if it is missing, non-numeric or non-finite."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v.strip()
    return "" if v is None else str(v).strip()


def _goal_value(g, name):
    """A goal cell from a label() result as a finite float, or None if unusable."""
    try:
        v = g.get(name)
    except AttributeError:
        return None
    return _finite(v)


def _features(rows, inputs):
    """Encode input cells as rows of floats in [0, 1]. Constant features are dropped.
    Returns one list per row (empty lists if nothing usable is left)."""
    n = len(rows)
    cols = []
    for c in inputs:
        raw = [r.get(c) for r in rows]
        vals = [_finite(v) for v in raw]
        present = [x for x in vals if x is not None]
        if present and len(present) >= NUMERIC_FRAC * n:
            if n >= 20 and len(set(present)) > ID_FRAC * n:
                continue  # identifier-like numeric column (ids, ranks)
            med = sorted(present)[len(present) // 2]
            cols.append([x if x is not None else med for x in vals])
            continue
        texts = [_text(v) for v in raw]
        levels = sorted(set(texts))
        if len(levels) > MAX_LEVELS:
            continue  # identifier-like text (names, urls, teams, nations)
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
        return [[] for _ in range(n)]
    return [list(t) for t in zip(*kept)]


def _scores(labelled, goals):
    """Score each benchmarked row with complete goal values, on the scale of step1.
    Returns {index: score in [0, 1]}; empty if no benchmarked row is usable."""
    oriented = {}
    for i, g in labelled.items():
        vec = []
        for name, sign in goals:
            x = _goal_value(g, name)
            if x is None:
                break
            vec.append(sign * x)
        else:
            oriented[i] = vec
    if not oriented:
        return {}

    k = len(goals)
    lo = [min(v[j] for v in oriented.values()) for j in range(k)]
    hi = [max(v[j] for v in oriented.values()) for j in range(k)]
    scores = {}
    for i, vec in oriented.items():
        total = 0.0
        for j in range(k):
            span = hi[j] - lo[j]
            if span > 0:
                total += (vec[j] - lo[j]) / span
        scores[i] = total / k
    return scores


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found while calling label() at most `budget` times.

    goal: optional goal column name (e.g. 'Penalties+'). If given, only that goal is
    optimised. Otherwise all goals are combined as in step1.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")

    rows, inputs, goals = _load(blind_csv)
    if not rows:
        raise ValueError("no candidate builds in blind_csv")

    if goal is not None:
        names = [c for c, _ in goals]
        if goal not in names:
            raise ValueError(f"{goal!r} is not a goal column of {blind_csv}")
        goals = [(c, s) for c, s in goals if c == goal]

    if not goals:
        # Nothing can be measured or optimised, so there is no basis for a choice.
        return 0

    n = len(rows)
    budget = min(budget, n)   # cannot benchmark more distinct rows than exist

    X = _features(rows, inputs)
    D = len(X[0]) if X and X[0] else 0
    scale = math.sqrt(D) if D > 0 else 1.0

    labelled = {}
    near = [math.inf] * n     # distance from each row to its nearest benchmarked row

    def run(i):
        if len(labelled) >= budget:
            raise RuntimeError("benchmark budget exhausted")
        labelled[i] = label(i)

    def add(i):
        run(i)
        xi = X[i]
        for j in range(n):
            d = math.dist(X[j], xi)
            if d < near[j]:
                near[j] = d

    rng = random.Random(SEED)

    # Phase 1: exploration by maximin (farthest-point) sampling.
    n_init = max(1, min(budget, int(round(budget * INIT_FRAC))))
    add(rng.randrange(n))
    while len(labelled) < n_init:
        pick = max((i for i in range(n) if i not in labelled),
                   key=lambda i: (near[i], -i))
        add(pick)

    # Phase 2: adaptive search.
    step = 0
    while len(labelled) < budget:
        kappa = KAPPA[step % len(KAPPA)]
        step += 1

        scores = _scores(labelled, goals)
        scored = list(scores)

        pick, pick_val = None, -math.inf
        for i in range(n):
            if i in labelled:
                continue
            if scored:
                nb = sorted((math.dist(X[i], X[j]), j) for j in scored)[:K_NEAR]
                wsum = 0.0
                acc = 0.0
                for d, j in nb:
                    w = 1.0 / (d + EPS)
                    wsum += w
                    acc += w * scores[j]
                pred = acc / wsum
            else:
                pred = 0.0
            val = pred + kappa * (near[i] / scale)
            if val > pick_val:
                pick, pick_val = i, val
        add(pick)

    # Final choice: best benchmarked row under the final normalisation.
    scores = _scores(labelled, goals)
    if scores:
        return max(scores, key=lambda i: (scores[i], -i))
    return min(labelled)

