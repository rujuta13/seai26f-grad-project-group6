"""Step 30: race our customer targeting and response guesser against simple rivals on test_csv.

race(train_csv, test_csv) returns {method: score}. Higher is better for every key.
A method whose score cannot be computed is reported as -inf.

Targeting race. The test customers form a blind pool, and each method may label
at most _BUDGET of them. The score is the chosen customer's mean goal score over
the complete test customers, each goal scaled over those customers (0 worst, 1 best).
  "ours"                step27.choose (scout-limited targeting search)
  "top_income"          label the _BUDGET customers with the highest INCOME and keep
                        the best labelled one (the "mail our richest customers" rival)
  "random_search"       label random customers, keep the best labelled one
  "off_the_shelf_model" ridge-guided labelling, as in step10

Response guesser race. Only the RESPONSE goal is predicted, on test rows where it
is finite. The score is minus the mean absolute error, scaled by the RESPONSE
range in train_csv.
  "response_guesser_neg_mae"  step22.guess median estimate ("ours")
  "response_ridge_neg_mae"    ridge regression fitted on train_csv
"""
import math
import os
import tempfile

from step1 import _load
from step10 import (
    _BUDGET,
    _Encoder,
    _best_of,
    _combined,
    _complete,
    _finite,
    _goal_specs,
    _is_input,
    _model_search,
    _neg_mae,
    _random_search,
    _ridge_preds,
    _spans,
    _write_blind,
)
from step22 import guess as _guess22
from step27 import choose as _choose27


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores (higher is better) for our targeting and response guesser and the rivals.

    Keys: "ours", "top_income", "random_search", "off_the_shelf_model"
    (targeting race, score = mean goal score of the chosen customer) and
    "response_guesser_neg_mae", "response_ridge_neg_mae" (RESPONSE prediction
    race, score = minus the mean scaled absolute error).
    """
    header, train = _load(train_csv)
    theader, test = _load(test_csv)
    specs = _goal_specs(theader)
    train_inputs = [c for c in header if _is_input(c)]
    test_inputs = [c for c in theader if _is_input(c)]
    goal_names = [c for c, _ in _goal_specs(header)]

    scores = {
        "ours": -math.inf,
        "top_income": -math.inf,
        "random_search": -math.inf,
        "off_the_shelf_model": -math.inf,
        "response_guesser_neg_mae": -math.inf,
        "response_ridge_neg_mae": -math.inf,
    }

    # Targeting race on the complete test customers.
    keep, vecs = _complete(test, specs)
    pool = [test[i] for i in keep]
    n = len(pool)
    if n > 0 and specs:
        pool_score = _combined(vecs)
        budget = min(_BUDGET, n)
        goal_cols = [c for c, _ in specs]

        def label(i):
            return {c: pool[i][c] for c in goal_cols}

        with tempfile.TemporaryDirectory() as tmp:
            blind = os.path.join(tmp, "blind.csv")
            _write_blind(blind, theader, pool)
            ours = _choose27(blind, label, budget=budget)
        scores["ours"] = float(pool_score[ours])

        if "INCOME" in theader:
            inc = [
                f if f is not None else -math.inf
                for f in (_finite(r.get("INCOME")) for r in pool)
            ]
            ranked = sorted(range(n), key=lambda i: (-inc[i], i))
            scores["top_income"] = float(
                pool_score[_best_of(ranked[:budget], vecs)]
            )

        scores["random_search"] = float(
            pool_score[_random_search(n, vecs, budget)]
        )

        enc = _Encoder(pool, test_inputs)
        X = [enc.encode(r) for r in pool]
        scores["off_the_shelf_model"] = float(
            pool_score[_model_search(X, vecs, budget)]
        )

    # Response guesser race: predict RESPONSE for every test customer from train.
    resp = [c for c in goal_names if c[:-1] == "RESPONSE"]
    if resp and train and test:
        goal = resp[0]
        spans = _spans(train, [goal])

        g_preds = []
        for r in test:
            row = {c: r[c] for c in train_inputs if c in r}
            est = _guess22(train_csv, row)
            g_preds.append([est[goal][0] if goal in est else None])
        scores["response_guesser_neg_mae"] = float(
            _neg_mae(g_preds, test, [goal], spans)
        )

        r_preds = _ridge_preds(train, test, train_inputs, [goal])
        scores["response_ridge_neg_mae"] = float(
            _neg_mae(r_preds, test, [goal], spans)
        )

    return scores

