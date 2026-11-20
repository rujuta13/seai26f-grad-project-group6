"""step7.py: choose a good flag set while benchmarking at most `budget` builds.

blind_csv holds the flags of every candidate build, with '?' in every goal
cell. label(i) benchmarks row i and returns its goals as a dict. It is called
at most `budget` times. The search has two phases:

  1. Explore: benchmark builds from every flag family (step1.families),
     round-robin, so each region of the flag space is sampled.
  2. Exploit: repeatedly benchmark the unmeasured build with the best predicted
     score. The prediction is the mean normalised score of its K nearest
     measured builds (as in step2), plus a bonus for being far from anything
     measured.

The answer is the measured build with the highest normalised goal sum. Such a
build is never dominated by another measured build, so it agrees with the
Pareto-front ranking of step1.best. Standard library only.
"""
import math
import random

from step1 import _load, _columns, families

EXPLORE_SHARE = 0.5   # fraction of the budget spent on exploration
K_NEIGHBOURS = 3      # measured neighbours used to predict an unmeasured build
EXPLORE_WEIGHT = 0.5  # weight of the distance bonus in the exploit phase
SEED = 1


def _as_goal(v):
    """A finite float, or None if the cell is missing or not numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sqd(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best build found, benchmarking at most `budget` builds.

    If `goal` names one goal column (with or without its +/- suffix), only that
    goal is optimised.
    """
    cols, rows = _load(blind_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if goal is not None:
        goals = [(c, s) for c, s in goals if c == goal or c[:-1] == goal]
        if not goals:
            raise ValueError(f"{goal!r} is not a goal column of {blind_csv}")
    if not goals:
        raise ValueError(f"{blind_csv} has no goal columns")

    n = len(rows)
    if n == 0:
        raise ValueError(f"{blind_csv} has no rows")
    if budget < 1:
        raise ValueError("budget must be at least 1")
    budget = min(budget, n)

    # Scale every numeric input to [0, 1] so no column dominates the distance.
    lo = {c: min(r[c] for r in rows) for c in inputs}
    hi = {c: max(r[c] for r in rows) for c in inputs}
    X = []
    for r in rows:
        x = []
        for c in inputs:
            span = hi[c] - lo[c]
            x.append((r[c] - lo[c]) / span if span > 0 else 0.0)
        X.append(x)

    fam = families(blind_csv)
    rng = random.Random(SEED)

    measured = {}  # row index -> oriented goal vector, or None if unusable
    calls = 0

    def measure(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("benchmark budget exhausted")
        calls += 1
        got = label(i)
        if not isinstance(got, dict):
            got = {}
        vec = []
        for c, sgn in goals:
            v = _as_goal(got.get(c))
            if v is None:
                vec = None
                break
            vec.append(sgn * v)
        measured[i] = vec

    def normaliser():
        """Score function: sum over goals of the position between the worst and
        best measured value (as in step1.best). None if nothing is usable."""
        usable = [v for v in measured.values() if v is not None]
        if not usable:
            return None
        g = len(goals)
        lo_g = [min(v[t] for v in usable) for t in range(g)]
        hi_g = [max(v[t] for v in usable) for t in range(g)]

        def score(v):
            s = 0.0
            for t in range(g):
                span = hi_g[t] - lo_g[t]
                s += (v[t] - lo_g[t]) / span if span > 0 else 1.0
            return s

        return score

    # Phase 1: explore, round-robin over flag families.
    n_explore = min(budget, max(1, round(budget * EXPLORE_SHARE)))
    groups = {}
    for i in range(n):
        groups.setdefault(fam[i], []).append(i)
    for members in groups.values():
        rng.shuffle(members)

    picks = []
    while len(picks) < n_explore:
        progressed = False
        for f in sorted(groups):
            if groups[f] and len(picks) < n_explore:
                picks.append(groups[f].pop())
                progressed = True
        if not progressed:
            break
    for i in picks:
        measure(i)

    # Phase 2: exploit, benchmark the unmeasured build with the best prediction.
    while calls < budget:
        unmeasured = [i for i in range(n) if i not in measured]
        if not unmeasured:
            break
        score = normaliser()
        if score is None:
            # Nothing usable measured yet: keep sampling at random.
            measure(rng.choice(unmeasured))
            continue

        known = [(i, score(v)) for i, v in measured.items() if v is not None]
        best_u, best_key = None, None
        for u in unmeasured:
            near = sorted((_sqd(X[u], X[i]), s) for i, s in known)[:K_NEIGHBOURS]
            pred = sum(s for _, s in near) / len(near)
            bonus = EXPLORE_WEIGHT * math.sqrt(near[0][0])
            key = pred + bonus
            if best_key is None or key > best_key:
                best_u, best_key = u, key
        measure(best_u)

    # Answer: the measured build with the highest normalised goal sum.
    score = normaliser()
    usable = [(i, v) for i, v in measured.items() if v is not None]
    if score is None or not usable:
        return min(measured)
    best_i, _ = max(usable, key=lambda iv: (score(iv[1]), -iv[0]))
    return int(best_i)

