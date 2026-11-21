"""Step 7: pick the best flag set within a benchmarking budget.

Standard library only. Loading, column splitting, input encoding, k-means,
goal normalisation, Pareto layers and the kNN predictor are reused from
steps 1 and 2. The goal cells of blind_csv are all '?' and are never read:
every goal value comes from the label callback.

Method, using at most `budget` calls to label:
  1. Explore: k-means on the encoded inputs with about half the budget of
     clusters. Benchmark the build nearest each cluster centre, so the first
     measurements cover the flag space.
  2. Exploit: benchmark, one at a time, the unmeasured build with the highest
     predicted score. Predictions use the inverse-distance kNN of step 2 on
     the measured builds, and scores are normalised with the bounds of the
     measured builds, as in step 1.

The answer is the best measured build: Pareto front first, then highest mean
normalised goal score, then lowest index. With `goal` set, only that goal is
optimised.
"""

import math
import random

from step1 import (
    SEED,
    _load,
    _columns,
    _input_matrix,
    _kmeans,
    _normalised_goals,
    _pareto_layers,
    _sqdist,
)
from step2 import _predict


def _select_goals(header, goal):
    """Input columns, and the goals to optimise (all goals, or just `goal`)."""
    inputs, goals = _columns(header)
    if goal is None:
        return inputs, goals
    chosen = [(name, sign) for name, sign in goals if name == goal]
    if not chosen:
        raise ValueError(f"{goal!r} is not a goal column of the file")
    return inputs, chosen


def _explore(points, inputs, k, rng):
    """Up to k distinct row indices spread over the encoded input space."""
    n = len(points)
    if not inputs:
        return rng.sample(range(n), k)
    dim = len(points[0])
    labels = _kmeans(points, k, rng)
    members = {}
    for i, lab in enumerate(labels):
        members.setdefault(lab, []).append(i)
    picks = []
    for lab in sorted(members):
        idx = members[lab]
        centre = [sum(points[i][j] for i in idx) / len(idx) for j in range(dim)]
        picks.append(min(idx, key=lambda i: (_sqdist(points[i], centre), i)))
    if len(picks) < k:
        taken = set(picks)
        rest = [i for i in range(n) if i not in taken]
        picks += rng.sample(rest, k - len(picks))
    return picks


def _bounds(targets, m):
    """Per goal, (lo, hi) of the signed measured values."""
    lo = [min(t[g] for t in targets) for g in range(m)]
    hi = [max(t[g] for t in targets) for g in range(m)]
    return lo, hi


def _mean_scaled(vals, lo, hi):
    """Mean of the goal values scaled to [0, 1] with bounds lo..hi.
    A goal with no spread counts as 0, as in step 1."""
    parts = []
    for v, a, b in zip(vals, lo, hi):
        span = b - a
        parts.append((v - a) / span if span > 0 else 0.0)
    return sum(parts) / len(parts)


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Row index of the best build found, using at most `budget` label calls.

    label(i) returns the goals of row i as a dict. If `goal` names one goal,
    only that goal is optimised.
    """
    if budget < 1:
        raise ValueError("budget must allow at least one benchmark")
    header, rows = _load(blind_csv)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no rows")
    inputs, goals = _select_goals(header, goal)
    if not goals:
        raise ValueError("the file has no goal columns (names ending in + or -)")
    m = len(goals)
    names = [name for name, _ in goals]

    points = _input_matrix(rows, inputs)
    rng = random.Random(SEED)
    limit = min(budget, n)
    measured = {}

    def measure(i):
        if i not in measured:
            if len(measured) >= budget:
                raise RuntimeError("benchmark budget exhausted")
            result = label(i)
            measured[i] = {name: float(result[name]) for name in names}

    def signed(i):
        return [sign * measured[i][name] for name, sign in goals]

    # Phase 1: spread the first measurements over the flag space.
    n_explore = min(limit, max(1, budget // 2))
    for i in _explore(points, inputs, n_explore, rng):
        measure(i)

    # Phase 2: benchmark the unmeasured build with the best predicted score.
    while len(measured) < limit:
        done = sorted(measured)
        lab_points = [points[i] for i in done]
        lab_targets = [signed(i) for i in done]
        lo, hi = _bounds(lab_targets, m)
        pick, pick_score = None, -math.inf
        for i in range(n):
            if i in measured:
                continue
            pred = _predict(points[i], lab_points, lab_targets)
            s = _mean_scaled(pred, lo, hi)
            if s > pick_score:
                pick, pick_score = i, s
        measure(pick)

    # Answer: best measured build, Pareto front first, then mean score.
    done = sorted(measured)
    norm = _normalised_goals([measured[i] for i in done], goals)
    layer_of = _pareto_layers(norm)
    score = [sum(p) / len(p) for p in norm]
    front = [k for k in range(len(done)) if layer_of[k] == 0]
    best_k = min(front, key=lambda k: (-score[k], done[k]))
    return done[best_k]

