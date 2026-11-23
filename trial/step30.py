"""step30.py: race our customer targeting and response guesser against simple rivals on test_csv.

Two client questions:
- Does our targeting beat simply mailing the highest-income customers? Our targeting is
  step17.choose over all goals with the same budget as the rivals. The rival mails the
  `budget` customers with the highest INCOME.
- Does our response guesser beat an off-the-shelf model? Our guesser is step2.guess for
  RESPONSE+. The rival is a default RandomForestRegressor fitted on the same encoded inputs.

All scores are higher-is-better, on the same scales as step10 and step20.
"""

import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _load, _encode
from step2 import guess
from step5 import _goal_scores
from step10 import _BUDGET, _write_blind, _random_search_score
from step17 import choose

_INCOME = "INCOME"
_RESP = "RESPONSE+"


def _income_top_score(full, rows, budget):
    """Best combined score among the `budget` rows with the highest INCOME."""
    income = []
    for r in rows:
        try:
            v = float(r[_INCOME])
            income.append(v if np.isfinite(v) else -np.inf)
        except (TypeError, ValueError):
            income.append(-np.inf)
    order = sorted(range(len(rows)), key=lambda i: (-income[i], i))
    return float(full[order[:budget]].max())


def _response_scores(train_csv, train_rows, test_rows, inputs):
    """Negated mean absolute error on RESPONSE+, scaled by its test spread, for
    our guesser (step2.guess) and an off-the-shelf random forest."""
    truth = np.array([float(r[_RESP]) for r in test_rows], dtype=float)
    spread = float(truth.std())
    if not spread > 0:
        spread = 1.0

    # Ours: step2.guess, one query per test customer.
    pred_ours = np.zeros(len(test_rows), dtype=float)
    for i, r in enumerate(test_rows):
        row = {c: r.get(c, "") for c in inputs}
        pred_ours[i] = float(guess(train_csv, row)[_RESP][0])
    ours = -float(np.mean(np.abs(pred_ours - truth)) / spread)

    # Off-the-shelf: default random forest on the same encoded inputs.
    if inputs:
        X_all = _encode(train_rows + test_rows, inputs)
        X_tr, X_te = X_all[:len(train_rows)], X_all[len(train_rows):]
    else:
        X_tr = np.zeros((len(train_rows), 1), dtype=float)
        X_te = np.zeros((len(test_rows), 1), dtype=float)
    y_tr = np.array([float(r[_RESP]) for r in train_rows], dtype=float)
    model = RandomForestRegressor(random_state=0, n_jobs=1)
    model.fit(X_tr, y_tr)
    pred_rf = np.asarray(model.predict(X_te), dtype=float).reshape(len(test_rows))
    rf = -float(np.mean(np.abs(pred_rf - truth)) / spread)

    return ours, rf


def race(train_csv, test_csv) -> dict[str, float]:
    """Score each method on test_csv. Higher is better.

    Keys: "ours" (step17.choose targeting), "top_income" (highest INCOME, when
    INCOME is an input), "random_search" (uniform sampling), and, when RESPONSE+
    is a goal in both files, "response_guesser_ours" (step2.guess) and
    "response_guesser_rf" (default random forest).
    """
    test_rows, inputs, goals = _load(test_csv)
    if not test_rows or not goals:
        return {}

    train_rows, _, train_goals = _load(train_csv)
    goal_names = [name for name, _ in goals]
    n = len(test_rows)
    budget = min(_BUDGET, n)

    # Shared pool and budget: every tuner-style method samples test_csv customers.
    full = _goal_scores(test_rows, goals)

    def label(i):
        return {name: float(test_rows[i][name]) for name in goal_names}

    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(test_csv, blind)
        out["ours"] = float(full[choose(blind, label, budget=budget)])

    if _INCOME in inputs:
        out["top_income"] = _income_top_score(full, test_rows, budget)
    out["random_search"] = _random_search_score(full, budget)

    train_goal_names = [name for name, _ in train_goals]
    if _RESP in goal_names and _RESP in train_goal_names and train_rows:
        r_ours, r_rf = _response_scores(train_csv, train_rows, test_rows, inputs)
        out["response_guesser_ours"] = r_ours
        out["response_guesser_rf"] = r_rf

    return out

