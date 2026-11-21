"""Step 10: race. Our tuner and guesser against stdlib-only rivals.

Tuner race (higher is better): each method gets BUDGET label calls on the
rows of test_csv and returns the build it picks. The score is that build's
mean normalised goal value over all test rows (0 = worst, 1 = best), so
"ours" is step 7's choose(). Rivals:
  - random_search: BUDGET distinct random builds, same answer rule.
  - hill_climb: Python's standard library ships no autotuner, so this
    stand-in uses only `random`: a nearest-neighbour local search with random
    restarts, ranked by the same answer rule.

Guesser race (higher is better): the score is the negative mean absolute
error, in units of each goal's training span, of predicted goal values on
test_csv. The guesser is step 2's guess(). Its rival is a ridge regression
fitted on the same encoded inputs. The standard library has no off-the-shelf
regression model, so the ridge fit is implemented here with the normal
equations and Gaussian elimination.

Scores in one dict are on different scales (tuner in [0, 1], guesser at most
0). Compare methods within each family only.
"""

import random

from step1 import (
    SEED,
    _load,
    _columns,
    _input_matrix,
    _normalised_goals,
    _pareto_layers,
    _sqdist,
)
from step2 import _model, guess
from step4 import _goal_bounds
from step7 import choose, _bounds as _measured_bounds, _mean_scaled


BUDGET = 30          # benchmark calls given to every tuner
RESTART_PROB = 0.2   # hill climber: chance of a random (not nearest) next pick
RIDGE = 1e-3         # ridge penalty of the off-the-shelf regression

TUNER_KEYS = ("ours", "random_search", "hill_climb")
GUESS_KEYS = ("guesser_ours", "guesser_ridge")


def _pick(measured, goals):
    """Step 7's answer rule on {row index: goal values}: Pareto front first,
    then highest mean normalised score, then lowest index."""
    done = sorted(measured)
    norm = _normalised_goals([measured[i] for i in done], goals)
    layer_of = _pareto_layers(norm)
    score = [sum(p) / len(p) for p in norm]
    front = [k for k in range(len(done)) if layer_of[k] == 0]
    best_k = min(front, key=lambda k: (-score[k], done[k]))
    return done[best_k]


def _hill_climb(points, label, goals, budget, rng):
    """Nearest-neighbour local search with random restarts.

    Each step moves to the unmeasured build nearest the best build so far
    (or, with probability RESTART_PROB, to a random unmeasured build).
    Returns {row index: goal values} for the measured builds."""
    n = len(points)
    m = len(goals)
    limit = min(budget, n)
    measured = {}

    def measure(i):
        if i not in measured:
            measured[i] = {name: float(v) for name, v in label(i).items()}

    def signed(i):
        return [sign * measured[i][name] for name, sign in goals]

    def incumbent():
        done = sorted(measured)
        lo, hi = _measured_bounds([signed(i) for i in done], m)
        return max(done, key=lambda i: (_mean_scaled(signed(i), lo, hi), -i))

    measure(rng.randrange(n))
    while len(measured) < limit:
        unmeasured = [i for i in range(n) if i not in measured]
        if rng.random() < RESTART_PROB:
            pick = rng.choice(unmeasured)
        else:
            cur = points[incumbent()]
            pick = min(unmeasured, key=lambda i: (_sqdist(points[i], cur), i))
        measure(pick)
    return measured


def _solve(A, B):
    """Solve A X = B by Gauss-Jordan elimination with partial pivoting.
    A is d x d, B is d x m; returns X as d x m."""
    d = len(A)
    M = [A[i][:] + B[i][:] for i in range(d)]
    for col in range(d):
        piv = max(range(col, d), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if p == 0.0:
            continue
        M[col] = [v / p for v in M[col]]
        for r in range(d):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[d:] for row in M]


def _ridge_fit(X, Y, lam):
    """Ridge coefficients W (d x m) for rows X (with bias column first) and
    targets Y. The bias is not penalised."""
    d = len(X[0])
    m = len(Y[0])
    A = [[0.0] * d for _ in range(d)]
    B = [[0.0] * m for _ in range(d)]
    for x, y in zip(X, Y):
        for a in range(d):
            xa = x[a]
            if xa:
                ra = A[a]
                for b in range(d):
                    ra[b] += xa * x[b]
                rb = B[a]
                for g in range(m):
                    rb[g] += xa * y[g]
    for a in range(1, d):
        A[a][a] += lam
    return _solve(A, B)


def _linear_predict(W, x):
    m = len(W[0])
    return [sum(xa * W[a][g] for a, xa in enumerate(x)) for g in range(m)]


def _guesser_scores(train_csv, rows, inputs, names):
    """Negative mean absolute error (in training spans) of step 2's guess
    and of the ridge rival, on the rows of the test file."""
    model = _model(train_csv)
    spans = [hi - lo if hi - lo > 0 else 1.0 for lo, hi in _goal_bounds(model)]
    total = len(rows) * len(names)

    err = 0.0
    for r in rows:
        probe = {c: r[c] for c in inputs}
        est = guess(train_csv, probe)
        for g, name in enumerate(names):
            err += abs(est[name][0] - r[name]) / spans[g]
    ours = -err / total

    # Encode train and test together so that training codes match
    # model["points"], as step 2 does.
    train_rows = model["rows"]
    k = len(train_rows)
    feats = _input_matrix(train_rows + rows, inputs)
    X = [[1.0] + x for x in feats[:k]]
    W = _ridge_fit(X, model["targets"], RIDGE)

    err = 0.0
    for x, r in zip(feats[k:], rows):
        pred = _linear_predict(W, [1.0] + x)
        for g, name in enumerate(names):
            err += abs(pred[g] - r[name]) / spans[g]
    ridge = -err / total

    return {"guesser_ours": ours, "guesser_ridge": ridge}


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores for our tuner and guesser and their rivals on test_csv.

    Tuner keys (mean normalised goal value of the build picked, higher is
    better): "ours", "random_search", "hill_climb".
    Guesser keys (negative mean normalised absolute error, higher is better):
    "guesser_ours", "guesser_ridge".
    """
    header, rows = _load(test_csv)
    inputs, goals = _columns(header)
    n = len(rows)
    names = [name for name, _ in goals]
    keys = TUNER_KEYS + GUESS_KEYS
    if n == 0 or not goals:
        return {k: 0.0 for k in keys}

    # True quality of every test build, normalised over the test file.
    truth = [sum(p) / len(p) for p in _normalised_goals(rows, goals)]

    def label(i):
        return {name: rows[i][name] for name in names}

    results = {}

    # Our tuner (step 7).
    results["ours"] = truth[choose(test_csv, label, budget=BUDGET)]

    # Random search with the same budget and answer rule.
    rng = random.Random(SEED)
    picks = rng.sample(range(n), min(BUDGET, n))
    results["random_search"] = truth[_pick({i: label(i) for i in picks}, goals)]

    # Stdlib-only hill climber.
    rng = random.Random(SEED)
    points = _input_matrix(rows, inputs)
    results["hill_climb"] = truth[
        _pick(_hill_climb(points, label, goals, BUDGET, rng), goals)
    ]

    # Guesser (step 2) against the ridge regression rival.
    results.update(_guesser_scores(train_csv, rows, inputs, names))

    return {k: float(results[k]) for k in keys}

