"""step30.py: race the targeting and response guesser against standard rivals on test_csv.

Targeting question ("does our targeting beat simply mailing our highest-income customers?"):
  Every method may measure at most `budget` test customers (budget = min(50, n)),
  then names one customer using the same Pareto-then-mean-rank rule as step10.
  ours        step27.choose (phased exploration and refinement)
  top_income  the `budget` customers with the highest INCOME
  random      `budget` customers drawn uniformly at random (fixed seed)
  Score: mean-goal-rank position of the named customer among all test customers,
  scaled to [0, 1]. Lower is better.

Response question ("does our response guesser beat an off-the-shelf model?"):
  ours_guess  step12.guess estimate of RESPONSE+ (k-nearest-neighbour median)
  rf_guess    random forest regressor on the same trait inputs
  Score: mean absolute error on RESPONSE+, divided by the training spread.
  Lower is better.

Both questions use trait inputs only, so rivals and our methods see the same information.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _classify, _goal_matrix, _load
from step4 import _encode_row, _encode_train
from step10 import _pick_measured, _tuner_score, _write_blind
from step11 import _trait_inputs
from step12 import guess
from step27 import choose

_BUDGET = 50        # test customers that may be measured by every targeting method
_SEED = 0           # fixed seed so the race is repeatable
_RF_TREES = 200
_INCOME = "INCOME"
_RESP = "RESPONSE+"


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores of our targeting and response guesser against standard rivals on test_csv.

    Keys: "ours", "top_income", "random" (targeting, see module docstring) and
    "ours_guess", "rf_guess" (response guessers). All scores are lower-is-better.
    """
    header, train_rows = _load(train_csv)
    test_header, test_rows = _load(test_csv)
    if test_header != header:
        raise ValueError("train_csv and test_csv must have the same header")
    if not train_rows or not test_rows:
        raise ValueError("train_csv and test_csv must both contain data rows")

    maximise, minimise, inputs = _classify(header)
    goal_cols = minimise + maximise
    if not goal_cols:
        raise ValueError("files have no goal columns")
    if _INCOME not in header:
        raise ValueError(f"column {_INCOME!r} is required for the income rival")
    if _RESP not in header or header.index(_RESP) not in goal_cols:
        raise ValueError(f"goal column {_RESP!r} is required for the response guesser")

    n = len(test_rows)
    budget = min(_BUDGET, n)
    goal_names = [header[c] for c in goal_cols]
    G = _goal_matrix(test_rows, maximise, minimise)

    scores: dict[str, float] = {}

    # ---- Targeting ----
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(header, test_rows, goal_cols, blind)

        def label(i):
            return {header[c]: float(test_rows[i][c]) for c in goal_cols}

        scores["ours"] = _tuner_score(G, choose(blind, label, budget=budget))

    income_col = header.index(_INCOME)
    income = np.array([float(r[income_col]) for r in test_rows], dtype=float)
    top_ids = np.argsort(-income, kind="stable")[:budget]
    scores["top_income"] = _tuner_score(G, _pick_measured(G, top_ids))

    rng = np.random.default_rng(_SEED)
    rand_ids = rng.choice(n, size=budget, replace=False)
    scores["random"] = _tuner_score(G, _pick_measured(G, rand_ids))

    # ---- Response guessing ----
    resp_col = header.index(_RESP)
    truth = np.array([float(r[resp_col]) for r in test_rows], dtype=float)
    train_resp = np.array([float(r[resp_col]) for r in train_rows], dtype=float)
    sd = float(np.std(train_resp))
    if sd == 0.0 or not np.isfinite(sd):
        sd = 1.0

    def mae(pred):
        return float(np.mean(np.abs(pred - truth))) / sd

    ours_pred = np.empty(n, dtype=float)
    for i, row in enumerate(test_rows):
        est = guess(train_csv, dict(zip(header, row)))
        ours_pred[i] = est[_RESP][0]
    scores["ours_guess"] = mae(ours_pred)

    traits = _trait_inputs(header, inputs)
    if traits:
        Xtrain, specs = _encode_train(train_rows, traits)
        F = np.array(
            [_encode_row(dict(zip(header, r)), header, traits, specs) for r in test_rows],
            dtype=float,
        )
    else:
        Xtrain = np.zeros((len(train_rows), 1), dtype=float)
        F = np.zeros((n, 1), dtype=float)

    forest = RandomForestRegressor(n_estimators=_RF_TREES, random_state=_SEED)
    forest.fit(Xtrain, train_resp)
    scores["rf_guess"] = mae(forest.predict(F))

    return scores

