"""Step 7: choose. Find a good flag set while benchmarking at most `budget` builds.

blind_csv has the inputs of every candidate build, with every goal cell set to '?'.
label(i) benchmarks build i and returns its goals as a dict. The budget is spent in
two phases:
  1. An exploration phase: maximin (farthest-point) sampling over the normalised
     inputs, so the first builds cover the input space.
  2. An adaptive phase: each new build is picked by an acquisition value,
     (predicted score from the nearest benchmarked builds) + KAPPA * (distance to
     the nearest benchmarked build). KAPPA cycles through exploit, exploit, explore.

Scores use the same scale as step1: each goal is min-max scaled over the benchmarked
builds (0 = worst, 1 = best) and averaged over goals. With `goal` set, only that goal
is scored. The returned index is the benchmarked build with the highest score.
"""
import math
import random

from step1 import _load, _encode

SEED = 1          # fixed seed so results are reproducible
INIT_FRAC = 0.5   # share of the budget spent on the exploration phase
K_NEAR = 3        # benchmarked neighbours used to predict an unbenchmarked build's score
KAPPA = (0.0, 0.0, 0.5)   # exploit, exploit, explore cycle for the adaptive phase


def _unit(X):
    """Min-max scale each input column to [0, 1] so no column dominates distances."""
    if not X or not X[0]:
        return X
    d = len(X[0])
    lo = [min(x[j] for x in X) for j in range(d)]
    hi = [max(x[j] for x in X) for j in range(d)]
    out = []
    for x in X:
        out.append([(x[j] - lo[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else 0.0
                    for j in range(d)])
    return out


def _dist(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _value(goal_dict, name):
    """A goal cell as a finite float, or None if missing or not a number."""
    try:
        v = float(goal_dict.get(name))
    except (AttributeError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _scores(labelled, goals):
    """Score each benchmarked build with complete goal values, on the scale of step1.

    Returns {index: score in [0, 1]}. Higher is better. Empty if nothing is usable.
    """
    oriented = {}
    for i, g in labelled.items():
        vec = []
        for name, sign in goals:
            v = _value(g, name)
            if v is None:
                break
            vec.append(sign * v)
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
    """Index of the best build found while calling label() at most `budget` times.

    goal: optional goal column name (e.g. 'Performance-' or 'Cpu-'). If given, only
    that goal is optimised. Otherwise all goals are combined as in step1.
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
        # Nothing can be measured or optimised; there is no basis for a choice.
        return 0

    n = len(rows)
    budget = min(budget, n)   # cannot benchmark more distinct builds than exist

    X = _unit(_encode(rows, inputs))
    D = len(X[0]) if X and X[0] else 0
    scale = math.sqrt(D) if D > 0 else 1.0

    labelled = {}
    calls = 0

    def run(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("benchmark budget exhausted")
        calls += 1
        labelled[i] = label(i)

    rng = random.Random(SEED)

    # Phase 1: exploration by maximin (farthest-point) sampling.
    n_init = max(1, min(budget, int(budget * INIT_FRAC)))
    run(rng.randrange(n))
    while len(labelled) < n_init:
        pick, pick_d = None, -1.0
        for i in range(n):
            if i in labelled:
                continue
            d = min(_dist(X[i], X[j]) for j in labelled)
            if d > pick_d:
                pick, pick_d = i, d
        run(pick)

    # Phase 2: adaptive search.
    step = 0
    while len(labelled) < budget:
        scores = _scores(labelled, goals)
        scored_ids = sorted(scores)
        kappa = KAPPA[step % len(KAPPA)]
        step += 1

        pick, pick_val = None, -math.inf
        for i in range(n):
            if i in labelled:
                continue
            near_all = min(_dist(X[i], X[j]) for j in labelled) / scale
            if scored_ids:
                nb = sorted((_dist(X[i], X[j]), j) for j in scored_ids)[:K_NEAR]
                wsum = sum(1.0 / (d + 1e-6) for d, _ in nb)
                pred = sum(scores[j] / (d + 1e-6) for d, j in nb) / wsum
            else:
                pred = 0.0
            val = pred + kappa * near_all
            if val > pick_val:
                pick, pick_val = i, val
        run(pick)

    # Final choice: best benchmarked build under the final normalisation.
    scores = _scores(labelled, goals)
    if scores:
        return max(scores, key=lambda i: (scores[i], -i))
    return min(labelled)

