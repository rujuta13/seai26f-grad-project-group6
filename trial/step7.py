"""step7.py - client question 8: choose a flag set within a benchmark budget.

choose(blind_csv, label, budget=30, goal=None) returns the index of the best
row found while calling label(i) at most `budget` times. blind_csv has the
same flag columns as the training data, but every goal cell is '?'.

Search, in two phases:
1. Coverage: k-means (step1._best_kmeans) splits the flag sets into
   budget // 2 clusters, and the flag set nearest each centroid is benchmarked
   first, so every family of flag sets gets a sample.
2. Refinement: the remaining calls alternate between
   - exploiting: the unbenchmarked flag set nearest the current best, and
   - exploring: the unbenchmarked flag set farthest from all benchmarked ones.

Goals: with goal=None all goals count. The answer is chosen among the
non-dominated benchmarked rows (step1._dominates), ties broken by the mean
goal score, where each goal is min-max scaled over the benchmarked rows.
With goal set to one goal, only that goal is optimised.

Reuses from step1: SEED, _best_kmeans, _dominates, _sqdist.
step1._load is not used because it cannot parse '?' goal cells.
"""
import csv
import random

from step1 import SEED, _best_kmeans, _dominates, _sqdist


def _parse(blind_csv):
    """Return (input_cols, goal_cols, X) without reading goal cells.

    goal_cols is a list of (name, sign): +1 maximise, -1 minimise.
    X[i] is the list of input values of row i.
    """
    with open(blind_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        input_cols, goal_cols = [], []
        for c in cols:
            if c.endswith("X"):
                continue
            if c.endswith("+"):
                goal_cols.append((c, +1))
            elif c.endswith("-"):
                goal_cols.append((c, -1))
            else:
                input_cols.append(c)
        X = [[float(row[c]) for c in input_cols] for row in reader]
    return input_cols, goal_cols, X


def _standardise(X, d):
    """Z-score each input column, as step2 does. Constant columns get std 1."""
    n = len(X)
    means = [sum(x[t] for x in X) / n for t in range(d)]
    stds = []
    for t in range(d):
        var = sum((x[t] - means[t]) ** 2 for x in X) / n
        s = var ** 0.5
        stds.append(s if s > 0 else 1.0)
    return [[(x[t] - means[t]) / stds[t] for t in range(d)] for x in X]


def _goal_indices(goal_cols, goal):
    """Indices of the goals to optimise. A goal may be named with or without
    its '+'/'-' suffix."""
    if goal is None:
        return list(range(len(goal_cols)))
    for g, (name, _) in enumerate(goal_cols):
        if name == goal:
            return [g]
    for g, (name, _) in enumerate(goal_cols):
        if name.rstrip("+-") == goal.rstrip("+-"):
            return [g]
    names = [name for name, _ in goal_cols]
    raise ValueError(f"no goal named {goal!r}; goals are {names}")


def _scores(sub):
    """Mean min-max scaled score per row. sub maps row -> list of
    maximisation-signed values. A goal with no spread scores 0.5 for all."""
    k = len(next(iter(sub.values())))
    total = {i: 0.0 for i in sub}
    for t in range(k):
        vs = [v[t] for v in sub.values()]
        lo, hi = min(vs), max(vs)
        span = hi - lo
        for i, v in sub.items():
            total[i] += (v[t] - lo) / span if span > 0 else 0.5
    return {i: s / k for i, s in total.items()}


def _pick(results, cols):
    """Best benchmarked row: non-dominated on the chosen goals, then highest
    mean score, then lowest index."""
    sub = {i: [v[g] for g in cols] for i, v in results.items()}
    front = [i for i in sub
             if not any(_dominates(sub[j], sub[i]) for j in sub if j != i)]
    score = _scores(sub)
    return max(front, key=lambda i: (score[i], -i))


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best flag set found while benchmarking at most `budget`
    rows. label(i) returns {goal column name: value} for row i. If `goal`
    names one goal, only that goal is optimised."""
    input_cols, goal_cols, X = _parse(blind_csv)
    n = len(X)
    if n == 0:
        raise ValueError("blind file has no data rows")
    if not goal_cols:
        raise ValueError("blind file has no goal columns")
    if budget < 1:
        raise ValueError("budget must be at least 1")

    cols = _goal_indices(goal_cols, goal)
    budget = min(budget, n)
    d = len(input_cols)
    Z = _standardise(X, d)

    results = {}  # row index -> maximisation-signed values of all goals

    def bench(i):
        if i in results:
            return
        if len(results) >= budget:
            raise RuntimeError("benchmark budget exhausted")
        vals = label(i)
        results[i] = [sign * float(vals[name]) for name, sign in goal_cols]

    # Phase 1: coverage. One benchmarked flag set per k-means cluster.
    rng = random.Random(SEED)
    explore_n = max(1, budget // 2)
    labels = _best_kmeans(Z, min(explore_n, n), rng)
    for c in sorted(set(labels)):
        if len(results) >= budget:
            break
        members = [i for i in range(n) if labels[i] == c]
        centre = [sum(Z[i][t] for i in members) / len(members)
                  for t in range(d)]
        bench(min(members, key=lambda i: (_sqdist(Z[i], centre), i)))

    # Phase 2: alternate exploiting the best with exploring the unknown.
    step = 0
    while len(results) < budget:
        unseen = [j for j in range(n) if j not in results]
        if not unseen:
            break
        best_i = _pick(results, cols)
        if step % 2 == 0:
            nxt = min(unseen,
                      key=lambda j: (_sqdist(Z[j], Z[best_i]), j))
        else:
            nxt = max(unseen,
                      key=lambda j: (min(_sqdist(Z[j], Z[e])
                                         for e in results), -j))
        bench(nxt)
        step += 1

    return int(_pick(results, cols))

