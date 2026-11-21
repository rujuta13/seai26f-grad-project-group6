"""step30.py - client question 30: race.

race(train_csv, test_csv) compares our methods with standard rivals on test_csv.

Targeting race. Each method may benchmark up to BUDGET test customers, then
returns the one it judges best. The score is that chosen row's quality on the
test set, in [0, 1], where 1 is the best test row and 0 the worst. Quality is
the step7 min-max mean over all goals, computed over every test row.
  ours           step7.choose (our targeting search)
  top_income     benchmark the BUDGET highest-INCOME rows, then pick the best
                 of them on the goals (the "mail the highest-income
                 customers" rival)
  random_search  BUDGET rows drawn uniformly at random (mean over REPEATS seeds)

Response race. Each method predicts RESPONSE+ of every test row from train_csv
alone. Score = 1 - MAE / training range of RESPONSE, so higher is better and
1 means exact predictions.
  ours_response_guesser  step2.guess (median of nearest training neighbours)
  linear_regression      ridge regression on standardised flags (step10)

Reused from earlier steps: SEED (step1), _load (step1), _model and guess
(step2), _scores and choose (step7), BUDGET, REPEATS, _blind_copy and
_fit_linear (step10).
"""
import os
import random
import tempfile

from step1 import SEED, _load
from step2 import _model, guess
from step7 import _scores, choose
from step10 import BUDGET, REPEATS, _blind_copy, _fit_linear


def _goal_pos(goal_cols, base):
    """Position of the goal whose name, without its suffix, is `base`."""
    for g, (name, _) in enumerate(goal_cols):
        if name.rstrip("+-") == base:
            return g
    raise ValueError(f"no {base} goal column")


def _targeting(test_csv):
    """Quality in [0, 1] of the row each targeting method picks on test_csv."""
    input_cols, goal_cols, X, Y = _load(test_csv)
    n = len(Y)
    if n == 0:
        raise ValueError("test file has no data rows")
    if "INCOME" not in input_cols:
        raise ValueError("test file has no INCOME input column")
    budget = min(BUDGET, n)
    t_inc = input_cols.index("INCOME")

    # quality of every test row over the whole test set (oracle view)
    truth = _scores({i: Y[i] for i in range(n)})

    def label_for(i):
        return {name: sign * Y[i][g] for g, (name, sign) in enumerate(goal_cols)}

    def pick_best(bench):
        # choice is made only from benchmarked rows, scaled over those rows
        sc = _scores(bench)
        return max(bench, key=lambda i: (sc[i], -i))

    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _blind_copy(test_csv, blind)
        ours_idx = choose(blind, label_for, budget=budget)

    # rival: benchmark the highest-income rows, then pick the best of them
    by_income = sorted(range(n), key=lambda i: (-X[i][t_inc], i))[:budget]
    income_idx = pick_best({i: Y[i] for i in by_income})

    # rival: uniform random sample of the budget, repeated over seeds
    rand_scores = []
    for r in range(REPEATS):
        rng = random.Random(SEED + r)
        sample = rng.sample(range(n), budget)
        rand_scores.append(truth[pick_best({i: Y[i] for i in sample})])

    return {
        "ours": truth[ours_idx],
        "top_income": truth[income_idx],
        "random_search": sum(rand_scores) / len(rand_scores),
    }


def _response(train_csv, test_csv):
    """Prediction quality in (-inf, 1] of the response guesser and of a ridge
    regression, both predicting RESPONSE from train_csv alone."""
    input_cols, train_goals, means, stds, Z, goal_vals = _model(train_csv)
    if not Z:
        raise ValueError("training file has no data rows")
    tin, test_goals, TX, TY = _load(test_csv)
    if tin != input_cols:
        raise ValueError("test and train input columns differ")

    gt = _goal_pos(train_goals, "RESPONSE")
    gs = _goal_pos(test_goals, "RESPONSE")
    train_name = train_goals[gt][0]

    n = len(TY)
    if n == 0:
        raise ValueError("test file has no data rows")

    # true raw response values of the test rows
    sign_s = test_goals[gs][1]
    truth = [sign_s * TY[i][gs] for i in range(n)]

    # training range of the response goal, used to normalise errors
    vals = [goal_vals[i][gt] for i in range(len(goal_vals))]
    span = max(vals) - min(vals)
    if span <= 0:
        span = 1.0

    # ours: step2 guesser (median of nearest training neighbours)
    ours_err = 0.0
    for i in range(n):
        row = {c: TX[i][t] for t, c in enumerate(tin)}
        est = guess(train_csv, row)
        ours_err += abs(est[train_name][0] - truth[i])

    # rival: ridge regression on standardised flags
    y = [goal_vals[i][gt] for i in range(len(Z))]
    ybar, w = _fit_linear(Z, y)
    lin_err = 0.0
    for i in range(n):
        z = [(TX[i][t] - means[t]) / stds[t] for t in range(len(tin))]
        pred = ybar + sum(a * b for a, b in zip(w, z))
        lin_err += abs(pred - truth[i])

    def score(err):
        return 1.0 - (err / n) / span

    return {
        "ours_response_guesser": score(ours_err),
        "linear_regression": score(lin_err),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare our targeting search and response guesser with standard rivals
    on test_csv. Returns {method: score}, higher is better, including "ours".

    Targeting scores ("ours", "top_income", "random_search") lie in [0, 1].
    Response scores are 1 minus the mean absolute error divided by the
    training range of RESPONSE, so 1 means exact.
    """
    result = _targeting(test_csv)
    result.update(_response(train_csv, test_csv))
    return {k: float(v) for k, v in result.items()}

