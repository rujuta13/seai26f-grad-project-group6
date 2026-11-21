"""step10.py - client question 10: race.

race(train_csv, test_csv) compares our methods with rivals on test_csv.

Tuning race. Each method may benchmark up to BUDGET test flag sets and then
returns the one it judges best. The score is that chosen row's quality on
the test set, in [0, 1], where 1 is the best test flag set and 0 the worst.
Quality is the step7 min-max mean over all goals, computed over every test row.
  ours           step7.choose
  random_search  BUDGET rows drawn uniformly at random (mean over REPEATS seeds)
  local_search   start at a random row, then repeatedly benchmark the unseen row
                 nearest the current best (mean over REPEATS seeds)

Prediction race. Each method predicts every goal of each test flag set from
train_csv alone. Score = 1 - mean over goals of (MAE / training range), so
higher is better and 1 means exact predictions.
  ours_guesser       step2.guess (median of the nearest training neighbours)
  linear_regression  ridge regression on standardised flags

The standard library has no tuner and no regression model, so the rivals are
written here with the standard library only.
"""
import csv
import os
import random
import tempfile

from step1 import SEED, _load, _sqdist
from step2 import _model, guess
from step7 import _scores, _standardise, choose

BUDGET = 30
REPEATS = 10
RIDGE = 1.0


def _blind_copy(test_csv, path):
    """Write a copy of test_csv with every goal cell replaced by '?'."""
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        goal = {c for c in fields if c.endswith("+") or c.endswith("-")}
        with open(path, "w", newline="") as out:
            w = csv.DictWriter(out, fieldnames=fields)
            w.writeheader()
            for row in reader:
                w.writerow({c: ("?" if c in goal else row[c]) for c in fields})


def _tuning(test_csv):
    """Quality in [0, 1] of the flag set each tuner picks on test_csv."""
    input_cols, goal_cols, X, Y = _load(test_csv)
    n = len(Y)
    if n == 0:
        raise ValueError("test file has no data rows")
    budget = min(BUDGET, n)
    Z = _standardise(X, len(input_cols))

    # quality of every test row over the whole test set (oracle view)
    truth = _scores({i: Y[i] for i in range(n)})

    def label_for(i):
        return {name: sign * Y[i][g] for g, (name, sign) in enumerate(goal_cols)}

    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _blind_copy(test_csv, blind)
        ours_idx = choose(blind, label_for, budget=budget)

    def pick_best(bench):
        # choice is made only from benchmarked rows, scaled over those rows
        sc = _scores(bench)
        return max(bench, key=lambda i: (sc[i], -i))

    rand_scores, local_scores = [], []
    for r in range(REPEATS):
        rng = random.Random(SEED + r)
        sample = rng.sample(range(n), budget)
        bench = {i: Y[i] for i in sample}
        rand_scores.append(truth[pick_best(bench)])

        rng = random.Random(SEED + r)
        start = rng.randrange(n)
        bench = {start: Y[start]}
        while len(bench) < budget:
            best_i = pick_best(bench)
            unseen = [j for j in range(n) if j not in bench]
            nxt = min(unseen, key=lambda j: (_sqdist(Z[j], Z[best_i]), j))
            bench[nxt] = Y[nxt]
        local_scores.append(truth[pick_best(bench)])

    return {
        "ours": truth[ours_idx],
        "random_search": sum(rand_scores) / len(rand_scores),
        "local_search": sum(local_scores) / len(local_scores),
    }


def _solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        if abs(piv) < 1e-12:
            continue
        for r in range(c + 1, n):
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))
        piv = M[r][r]
        x[r] = s / piv if abs(piv) >= 1e-12 else 0.0
    return x


def _fit_linear(Z, y, lam=RIDGE):
    """Ridge regression on standardised features Z for targets y.
    Returns (mean of y, weights)."""
    n, d = len(Z), len(Z[0]) if Z else 0
    ybar = sum(y) / n
    A = [[sum(Z[i][a] * Z[i][b] for i in range(n)) + (lam if a == b else 0.0)
          for b in range(d)] for a in range(d)]
    rhs = [sum(Z[i][a] * (y[i] - ybar) for i in range(n)) for a in range(d)]
    return ybar, _solve(A, rhs)


def _predicting(train_csv, test_csv):
    """Prediction quality in (-inf, 1] of the guesser and the linear model."""
    train_inputs, train_goals, means, stds, Z, goal_vals = _model(train_csv)
    if not Z:
        raise ValueError("training file has no data rows")
    tin, tgoal, TX, TY = _load(test_csv)
    if tin != train_inputs:
        raise ValueError("test and train input columns differ")
    if [name for name, _ in tgoal] != [name for name, _ in train_goals]:
        raise ValueError("test and train goal columns differ")

    n, m = len(TY), len(train_goals)
    if n == 0:
        raise ValueError("test file has no data rows")

    # true raw goal values of the test rows
    truth = [[sign * TY[i][g] for g, (_, sign) in enumerate(tgoal)]
             for i in range(n)]

    # per-goal training range, used to normalise errors
    spans = []
    for g in range(m):
        vals = [goal_vals[i][g] for i in range(len(goal_vals))]
        s = max(vals) - min(vals)
        spans.append(s if s > 0 else 1.0)

    # ours: step2 guesser
    ours_err = [0.0] * m
    for i in range(n):
        row = {c: TX[i][t] for t, c in enumerate(tin)}
        est = guess(train_csv, row)
        for g, (name, _) in enumerate(train_goals):
            ours_err[g] += abs(est[name][0] - truth[i][g])

    # rival: ridge regression, one model per goal
    fits = [_fit_linear(Z, [goal_vals[i][g] for i in range(len(Z))])
            for g in range(m)]
    lin_err = [0.0] * m
    for i in range(n):
        z = [(TX[i][t] - means[t]) / stds[t] for t in range(len(tin))]
        for g in range(m):
            ybar, w = fits[g]
            pred = ybar + sum(a * b for a, b in zip(w, z))
            lin_err[g] += abs(pred - truth[i][g])

    def score(err):
        return 1.0 - sum(err[g] / n / spans[g] for g in range(m)) / m

    return {
        "ours_guesser": score(ours_err),
        "linear_regression": score(lin_err),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare our tuner, guesser with rivals on test_csv.

    Returns {method: score}, higher is better. Tuner scores ("ours",
    "random_search", "local_search") lie in [0, 1]. Prediction scores
    ("ours_guesser", "linear_regression") are 1 minus the mean normalised
    absolute error, so 1 means exact.
    """
    result = _tuning(test_csv)
    result.update(_predicting(train_csv, test_csv))
    return {k: float(v) for k, v in result.items()}

