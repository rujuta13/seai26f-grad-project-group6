"""Step 30: race. Targeting against mailing the highest-income customers, and the
response guesser against an off-the-shelf model.

Standard library only. Reuses:
  - step1: loading, column splitting, goal normalisation and input encoding.
  - step2: the cached training model.
  - step4: the goal bounds used to scale errors.
  - step10: the answer rule (_pick), the stdlib hill climber, the ridge
    regression rival (_ridge_fit, _linear_predict, RIDGE) and BUDGET.
  - step22: the clipped response guesser (guess).
  - step27: the targeting (choose).

Scores (higher is better):
  - ours: mean normalised goal value of the build that step 27's choose()
    picks on test_csv (0 = worst, 1 = best over the test file).
  - top_income: the same score for mailing the BUDGET highest-income
    customers. Those customers are benchmarked and the best of them is
    picked by the same answer rule, so the budget is matched.
  - random_search, hill_climb: rival searches with the same budget and the
    same answer rule.
  - guesser_ours: negative mean absolute error, in training spans, of step
    22's guess() on RESPONSE+.
  - guesser_ridge: the same score for a ridge regression fitted on the
    training inputs for RESPONSE+.

Tuner scores lie in [0, 1] and guesser scores are at most 0. Compare methods
within each family only.
"""

import random

from step1 import SEED, _load, _columns, _input_matrix, _normalised_goals
from step2 import _model
from step4 import _goal_bounds
from step10 import BUDGET, RIDGE, _pick, _hill_climb, _ridge_fit, _linear_predict
from step22 import guess as _guess_response
from step27 import choose


INCOME = "INCOME"
RESPONSE_GOAL = "RESPONSE+"

TUNER_KEYS = ("ours", "top_income", "random_search", "hill_climb")
GUESS_KEYS = ("guesser_ours", "guesser_ridge")


def _top_income(rows, count):
    """Row indices of the `count` rows with the highest INCOME (ties by index)."""
    order = sorted(range(len(rows)), key=lambda i: (-float(rows[i][INCOME]), i))
    return order[:count]


def _response_scores(train_csv, rows):
    """Negative mean absolute error (in training spans) on RESPONSE+, for step
    22's guess() and for the ridge rival."""
    zeros = {"guesser_ours": 0.0, "guesser_ridge": 0.0}
    model = _model(train_csv)
    goal_names = [name for name, _ in model["goals"]]
    if (
        RESPONSE_GOAL not in goal_names
        or not rows
        or RESPONSE_GOAL not in rows[0]
        or not model["targets"]
    ):
        return zeros

    g = goal_names.index(RESPONSE_GOAL)
    inputs = model["inputs"]
    lo, hi = _goal_bounds(model)[g]
    span = hi - lo if hi - lo > 0 else 1.0
    n = len(rows)

    # Our guesser (step 22): step 2's estimator with calibrated, clipped ranges.
    err = 0.0
    for r in rows:
        est = _guess_response(train_csv, {c: r[c] for c in inputs})
        err += abs(est[RESPONSE_GOAL][0] - r[RESPONSE_GOAL]) / span
    ours = -err / n

    # Ridge rival, fitted on the training rows only. Train and test are encoded
    # together so that training codes match model["points"], as in step 2.
    train_rows = model["rows"]
    k = len(train_rows)
    feats = _input_matrix(train_rows + rows, inputs)
    X = [[1.0] + x for x in feats[:k]]
    Y = [[t[g]] for t in model["targets"]]
    W = _ridge_fit(X, Y, RIDGE)

    err = 0.0
    for x, r in zip(feats[k:], rows):
        pred = _linear_predict(W, [1.0] + x)[0]
        err += abs(pred - r[RESPONSE_GOAL]) / span
    ridge = -err / n

    return {"guesser_ours": ours, "guesser_ridge": ridge}


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores for our targeting and response guesser and their rivals on test_csv.

    Tuner keys (mean normalised goal value of the build picked, higher is
    better): "ours" (step 27 choose), "top_income" (best of the BUDGET
    highest-income customers), "random_search", "hill_climb".
    Guesser keys (negative mean normalised absolute error, higher is better):
    "guesser_ours" (step 22 guess on RESPONSE+), "guesser_ridge".
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

    # Our targeting (step 27).
    results["ours"] = truth[choose(test_csv, label, budget=BUDGET)]

    # Mailing the highest-income customers, with the same budget and answer rule.
    # Without an INCOME column this baseline cannot be formed, so it scores 0.0.
    if INCOME in header:
        top = _top_income(rows, min(BUDGET, n))
        results["top_income"] = truth[_pick({i: label(i) for i in top}, goals)]
    else:
        results["top_income"] = 0.0

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

    # Response guesser (step 22) against the ridge regression rival.
    results.update(_response_scores(train_csv, rows))

    return {k: float(results[k]) for k in keys}

