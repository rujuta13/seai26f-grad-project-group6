"""step10.py: race our tuner and guesser against simple rivals.

race(train_csv, test_csv) returns {method: score}. Higher is better for every method.

Tuner race. test_csv is the pool of candidate builds. The tuners see only the
flags; a goal is revealed only when a build is benchmarked. Each method may
benchmark at most BUDGET builds.
  - ours: step7.choose.
  - random_search: benchmark BUDGET random builds and keep the best measured.
  - hill_climb: greedy local search in flag space. After each benchmark, the
    next build is the unmeasured one with the fewest flag differences from the
    best build measured so far.
  The score of a chosen build is its goal sum over the test pool, with each goal
  scaled to [0, 1] (1 = best on every goal), averaged over goals and SEEDS.

Guesser race. Models are trained on train_csv and tested on test_csv.
  - guess_ours: step2.guess, the median goal of the nearest flag sets.
  - guess_linear: ordinary least squares on the numeric flags, solved with the
    normal equations and a tiny ridge term.
  The score is the mean R^2 over goals on test_csv (1 = perfect, 0 = no better
  than predicting the test mean).

The standard library has no tuner and no ML model. The hill climber and the
least-squares fit are therefore written here with the standard library only.
Standard library only.
"""
import csv
import math
import os
import random
import tempfile

import step7
from step1 import _load, _columns
from step2 import guess

BUDGET = 30            # benchmarks allowed per tuner per seed
SEEDS = (1, 2, 3, 4, 5)
RIDGE = 1e-6           # tiny ridge term on the linear model's non-intercept weights


# ---------------------------------------------------------------- shared helpers

def _oriented(d, goals):
    """Goal vector of a dict or row, oriented so higher is better; None if unusable."""
    vec = []
    for c, sgn in goals:
        try:
            x = float(d.get(c))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        vec.append(sgn * x)
    return vec


def _quality(V, g):
    """Per-vector score in [0, 1]: the mean position of each goal between the
    worst and best value in V (1 = best on every goal). None entries score 0."""
    usable = [v for v in V if v is not None]
    if not usable:
        return [0.0] * len(V)
    lo = [min(v[t] for v in usable) for t in range(g)]
    hi = [max(v[t] for v in usable) for t in range(g)]
    out = []
    for v in V:
        if v is None:
            out.append(0.0)
            continue
        s = 0.0
        for t in range(g):
            span = hi[t] - lo[t]
            s += (v[t] - lo[t]) / span if span > 0 else 1.0
        out.append(s / g)
    return out


def _best_index(seen, g):
    """Index of the best measured build (normalised over measured builds only),
    or None if nothing usable has been measured."""
    items = [(i, v) for i, v in sorted(seen.items()) if v is not None]
    if not items:
        return None
    s = _quality([v for _, v in items], g)
    k = max(range(len(items)), key=lambda k: (s[k], -items[k][0]))
    return items[k][0]


def _chosen_quality(seen, q, g):
    best = _best_index(seen, g)
    return q[best] if best is not None else 0.0


class _Bench:
    """Benchmarks builds through label, at most `budget` times, and remembers them."""

    def __init__(self, label, goals, budget):
        self._label = label
        self._goals = goals
        self._budget = budget
        self.seen = {}

    def __call__(self, i):
        if i not in self.seen:
            if len(self.seen) >= self._budget:
                raise RuntimeError("benchmark budget exhausted")
            self.seen[i] = _oriented(self._label(i), self._goals)
        return self.seen[i]


def _write_blind(cols, rows, goal_set, path):
    """Write rows with every goal cell replaced by '?' (a blind candidate pool)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("?" if c in goal_set else r[c]) for c in cols})


# ---------------------------------------------------------------- tuners

def _run_ours(blind, label, seed):
    """step7.choose with its random seed set to `seed` (restored afterwards)."""
    old = step7.SEED
    step7.SEED = seed
    try:
        return step7.choose(blind, label, budget=BUDGET)
    finally:
        step7.SEED = old


def _random_search(n, bench, seed, budget):
    rng = random.Random(seed)
    for i in rng.sample(range(n), budget):
        bench(i)


def _hill_climb(n, flags, bench, g, seed, budget):
    """Greedy local search: always benchmark the unmeasured build closest in
    flags to the best build measured so far (random while nothing is usable)."""
    rng = random.Random(seed)
    bench(rng.randrange(n))
    while len(bench.seen) < budget:
        cur = _best_index(bench.seen, g)
        unseen = [i for i in range(n) if i not in bench.seen]
        if cur is None:
            pick = rng.choice(unseen)
        else:
            dist = {i: sum(a != b for a, b in zip(flags[i], flags[cur]))
                    for i in unseen}
            m = min(dist.values())
            pick = rng.choice([i for i in unseen if dist[i] == m])
        bench(pick)


def _race_tuners(cols, rows, goals, flag_cols):
    names = [c for c, _ in goals]
    g = len(goals)
    n = len(rows)
    budget = min(BUDGET, n)
    q = _quality([_oriented(r, goals) for r in rows], g)
    flags = [tuple(r.get(c) for c in flag_cols) for r in rows]

    def label(i):
        return {c: rows[i][c] for c in names}

    totals = {"ours": 0.0, "random_search": 0.0, "hill_climb": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(cols, rows, set(names), blind)
        for seed in SEEDS:
            totals["ours"] += q[_run_ours(blind, label, seed)]

            bench = _Bench(label, goals, budget)
            _random_search(n, bench, seed, budget)
            totals["random_search"] += _chosen_quality(bench.seen, q, g)

            bench = _Bench(label, goals, budget)
            _hill_climb(n, flags, bench, g, seed, budget)
            totals["hill_climb"] += _chosen_quality(bench.seen, q, g)
    return {k: v / len(SEEDS) for k, v in totals.items()}


# ---------------------------------------------------------------- guessers

def _solve(A, b):
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        M[col] = [x / p for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [a - f * c for a, c in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def _fit_one(X, y):
    """Least-squares weights for rows X (each starting with 1.0) and targets y."""
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for x, t in zip(X, y):
        for i in range(d):
            b[i] += x[i] * t
            for j in range(d):
                A[i][j] += x[i] * x[j]
    for i in range(1, d):
        A[i][i] += RIDGE
    return _solve(A, b)


def _predict(w, x):
    return sum(a * b for a, b in zip(w, x))


def _r2(pairs):
    """R^2 of (truth, prediction) pairs."""
    if len(pairs) < 2:
        return 0.0
    ys = [y for y, _ in pairs]
    mean = sum(ys) / len(ys)
    sst = sum((y - mean) ** 2 for y in ys)
    sse = sum((y - p) ** 2 for y, p in pairs)
    if sst <= 0:
        return 1.0 if sse == 0 else 0.0
    return 1.0 - sse / sst


def _mean_r2(pairs_by_goal):
    scores = [_r2(p) for p in pairs_by_goal.values() if p]
    return sum(scores) / len(scores) if scores else 0.0


def _race_guessers(train_csv, rows, goal_names):
    tcols, trows = _load(train_csv)
    inputs, maxi, mini = _columns(tcols, trows)
    train_goals = set(maxi + mini)
    common = [g for g in goal_names if g in train_goals]

    # Inputs are numeric in every training row (step1._columns guarantees it).
    X_all = [[1.0] + [float(r[c]) for c in inputs] for r in trows]
    models = {}
    for g in common:
        pts = [(x, float(r[g])) for x, r in zip(X_all, trows)
               if isinstance(r[g], float) and math.isfinite(r[g])]
        if pts:
            models[g] = _fit_one([x for x, _ in pts], [y for _, y in pts])

    guess_pairs = {g: [] for g in common}
    lin_pairs = {g: [] for g in common}
    for r in rows:
        flags = {c: r.get(c) for c in inputs}
        pred = guess(train_csv, flags)
        x = [1.0] + [float(r[c]) if isinstance(r.get(c), float) else 0.0
                     for c in inputs]
        for g in common:
            y = r.get(g)
            if not (isinstance(y, float) and math.isfinite(y)):
                continue
            guess_pairs[g].append((y, pred[g][0]))
            w = models.get(g)
            lin_pairs[g].append((y, _predict(w, x) if w is not None else 0.0))

    return {"guess_ours": _mean_r2(guess_pairs),
            "guess_linear": _mean_r2(lin_pairs)}


# ---------------------------------------------------------------- entry point

def race(train_csv, test_csv) -> dict[str, float]:
    """Score our tuner (key "ours") and guesser against rivals on test_csv.

    Tuner scores lie in [0, 1]; guesser scores are mean R^2 (at most 1).
    Higher is better for every key.
    """
    cols, rows = _load(test_csv)
    _, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals or not rows:
        raise ValueError(f"{test_csv} needs goal columns and rows")
    flag_cols = [c for c in cols if not c.endswith(("X", "+", "-"))]

    out = {}
    out.update(_race_tuners(cols, rows, goals, flag_cols))
    out.update(_race_guessers(train_csv, rows, [c for c, _ in goals]))
    return {k: float(v) for k, v in out.items()}

