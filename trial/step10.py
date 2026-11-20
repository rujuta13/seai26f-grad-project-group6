"""step10.py: race. Race our tuner (step7) against a random search and a
stdlib-only local search on test_csv, and race our guesser (step2) against a
ridge regression written in pure Python.

The standard library ships no hyperparameter tuner and no machine-learning
model, so the rivals below are stand-ins built from stdlib pieces. They are
labelled as stand-ins in the keys and comments. Only the standard library is used.

Score scales: the tuner keys score the chosen build's true goals on a [0, 1]
scale (higher is better). The guesser keys score mean R^2 over the goals
(higher is better, can be negative). Compare within each group, not across them.
"""
from __future__ import annotations

import math
import random

from step1 import _SEED, _columns, _dist2, _features, _num, _read
from step2 import _fill, guess
from step4 import _encode, _fit, _objectives
from step5 import _scores
from step7 import _leader, _signed, choose

BUDGET = 30         # label calls allowed to each tuner
LOCAL_POOL = 5      # local search picks among this many nearest unmeasured builds
RIDGE = 1e-3        # ridge penalty for the linear rival


def _goal_values(rows: list[dict], i: int, goals: list[str]) -> dict:
    """Goal cells of row i, as the label function must return them."""
    return {g: rows[i].get(g) for g in goals}


def _random_search(rows: list[dict], goals: list[str], sign: list[float],
                   budget: int) -> int:
    """Measure a random sample of builds and return the leader."""
    rng = random.Random(_SEED)
    obs = {}
    for i in rng.sample(range(len(rows)), min(budget, len(rows))):
        obs[i] = _signed(_goal_values(rows, i, goals), goals, sign)
    return _leader(obs, len(goals))


def _local_search(rows: list[dict], feats: list[list[float]], goals: list[str],
                  sign: list[float], budget: int) -> int:
    """Stdlib stand-in for a tuner: a stochastic local search. Each step measures
    a random one of the LOCAL_POOL unmeasured builds nearest to the current leader."""
    rng = random.Random(_SEED + 1)
    n = len(rows)
    m = len(goals)
    limit = min(budget, n)
    obs = {}

    def measure(i: int) -> None:
        obs[i] = _signed(_goal_values(rows, i, goals), goals, sign)

    measure(rng.randrange(n))
    while len(obs) < limit:
        rest = [i for i in range(n) if i not in obs]
        if not rest:
            break
        best = _leader(obs, m)
        near = sorted(rest, key=lambda i: (_dist2(feats[i], feats[best]), i))
        measure(rng.choice(near[:LOCAL_POOL]))
    return _leader(obs, m)


def _r2(pairs: list[tuple[float, float]]) -> float | None:
    """Coefficient of determination for (truth, prediction) pairs, or None if
    the truth has no spread to explain."""
    if len(pairs) < 2:
        return None
    ts = [t for t, _ in pairs]
    mean_t = sum(ts) / len(ts)
    ss_tot = sum((t - mean_t) ** 2 for t in ts)
    if ss_tot <= 0.0:
        return None
    ss_res = sum((t - p) ** 2 for t, p in pairs)
    return 1.0 - ss_res / ss_tot


def _model_score(truth: dict[str, list], preds: dict[str, list[float]],
                 fallback: dict[str, float]) -> float:
    """Mean R^2 over goals. Non-finite predictions fall back to the training mean."""
    vals = []
    for g, ts in truth.items():
        ps = preds.get(g)
        if ps is None:
            continue
        pairs = []
        for t, p in zip(ts, ps):
            if t is None:
                continue
            if not math.isfinite(p):
                p = fallback.get(g, 0.0)
            pairs.append((t, p))
        r = _r2(pairs)
        if r is not None:
            vals.append(r)
    return sum(vals) / len(vals) if vals else 0.0


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pivot = M[c][c]
        if abs(pivot) < 1e-12:
            continue
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / pivot
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0 for i in range(n)]


def _ridge(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Ridge regression weights via the regularised normal equations."""
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for x, t in zip(X, y):
        for i in range(d):
            b[i] += x[i] * t
            xi = x[i]
            row = A[i]
            for j in range(d):
                row[j] += xi * x[j]
    for i in range(d):
        A[i][i] += lam
    return _solve(A, b)


def _linear_preds(train: list[dict], inputs: list[str], test: list[dict],
                  goals: list[str]) -> dict[str, list[float]]:
    """Stdlib stand-in for an off-the-shelf model: one ridge regression per goal
    on encoded inputs (bias term first). Goals with fewer than two training
    values are left out."""
    spec = _fit(train, inputs)
    X_train = [[1.0] + _encode(r, inputs, spec) for r in train]
    X_test = [[1.0] + _encode(_fill(r, train, inputs), inputs, spec) for r in test]
    out: dict[str, list[float]] = {}
    for g in goals:
        xs, ys = [], []
        for x, r in zip(X_train, train):
            v = _num(r.get(g))
            if v is not None:
                xs.append(x)
                ys.append(v)
        if len(ys) < 2:
            continue
        w = _ridge(xs, ys, RIDGE)
        out[g] = [sum(wi * xi for wi, xi in zip(w, x)) for x in X_test]
    return out


def race(train_csv, test_csv) -> dict[str, float]:
    """Race the methods on test_csv.

    Tuner keys (score in [0, 1], higher is better; the mean normalised goal
    score of the build each method picks, using its true test goals):
      "ours"           step7.choose with BUDGET label calls
      "random_search"  random sample of BUDGET builds, same leader rule
      "local_search"   stdlib stand-in for a tuner (the stdlib has none)

    Guesser keys (mean R^2 over goals, higher is better, can be negative):
      "ours_guesser"   step2.guess estimates, one call per test row
      "linear_model"   stdlib stand-in for an off-the-shelf model (ridge regression)
    """
    header, test = _read(str(test_csv))
    _, gmax, gmin = _columns(header)
    goals = gmax + gmin
    sign = [1.0] * len(gmax) + [-1.0] * len(gmin)
    names = ["ours", "random_search", "local_search", "ours_guesser", "linear_model"]
    n = len(test)
    if n == 0 or not goals:
        return {k: 0.0 for k in names}

    train_header, train = _read(str(train_csv))
    inputs, _, _ = _columns(train_header)

    # Tuner race: each method returns an index into test; score it on true goals.
    obj = [_objectives(r, gmax, gmin) for r in test]
    quality = _scores(obj, len(goals))
    budget = min(BUDGET, n)

    def label(i: int) -> dict:
        return _goal_values(test, i, goals)

    picks = {
        "ours": choose(test_csv, label, budget=budget),
        "random_search": _random_search(test, goals, sign, budget),
        "local_search": _local_search(test, _features(test, gmax and _columns(header)[0]),
                                      goals, sign, budget),
    }
    out: dict[str, float] = {k: float(quality[i]) for k, i in picks.items()}

    # Guesser race: predict each test row's goals and score by R^2.
    truth = {g: [_num(r.get(g)) for r in test] for g in goals}
    train_mean = {}
    for g in goals:
        vals = [v for v in (_num(r.get(g)) for r in train) if v is not None]
        train_mean[g] = sum(vals) / len(vals) if vals else 0.0

    guess_preds: dict[str, list[float]] = {g: [] for g in goals}
    for row in test:
        est = guess(train_csv, row)
        for g in goals:
            guess_preds[g].append(est[g][0] if g in est else math.nan)

    out["ours_guesser"] = float(_model_score(truth, guess_preds, train_mean))
    linear = _linear_preds(train, inputs, test, goals)
    out["linear_model"] = float(_model_score(truth, linear, train_mean))

    return out

