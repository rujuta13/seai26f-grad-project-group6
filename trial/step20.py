"""step20.py: race our shortlist and penalty guesser against simple rivals on test_csv.

Two client questions:
- Does our shortlist beat a simple "top overall rating" list? The shortlist is
  step17.choose (our tuner) with the same budget as the rivals. The rival picks
  the `budget` builds with the highest OVR.
- Does our penalty guesser beat an off-the-shelf model? Our guesser is
  step2.guess for the Penalties+ goal. The rival is a default
  RandomForestRegressor fitted on the same encoded inputs.

All scores are higher-is-better, on the same scales as step10.
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

_PEN = "Penalties+"
_OVR = "OVR"


def _top_overall_score(full, rows, budget):
    """Best combined score among the `budget` rows with the highest OVR."""
    ovr = []
    for r in rows:
        try:
            ovr.append(float(r[_OVR]))
        except (TypeError, ValueError):
            ovr.append(-np.inf)
    order = sorted(range(len(rows)), key=lambda i: (-ovr[i], i))
    return float(full[order[:budget]].max())


def _penalty_scores(train_csv, train_rows, test_rows, inputs):
    """Negated mean absolute error on Penalties+, scaled by its test spread, for
    our guesser (step2.guess) and an off-the-shelf random forest."""
    truth = np.array([float(r[_PEN]) for r in test_rows], dtype=float)
    spread = float(truth.std())
    if not spread > 0:
        spread = 1.0

    # Ours: step2.guess, one query per test build.
    pred_ours = np.zeros(len(test_rows), dtype=float)
    for i, r in enumerate(test_rows):
        row = {c: r.get(c, "") for c in inputs}
        pred_ours[i] = float(guess(train_csv, row)[_PEN][0])
    ours = -float(np.mean(np.abs(pred_ours - truth)) / spread)

    # Off-the-shelf: default random forest on the same encoded inputs.
    if inputs:
        X_all = _encode(train_rows + test_rows, inputs)
        X_tr, X_te = X_all[:len(train_rows)], X_all[len(train_rows):]
    else:
        X_tr = np.zeros((len(train_rows), 1), dtype=float)
        X_te = np.zeros((len(test_rows), 1), dtype=float)
    y_tr = np.array([float(r[_PEN]) for r in train_rows], dtype=float)
    model = RandomForestRegressor(random_state=0, n_jobs=1)
    model.fit(X_tr, y_tr)
    pred_rf = np.asarray(model.predict(X_te), dtype=float).reshape(len(test_rows))
    rf = -float(np.mean(np.abs(pred_rf - truth)) / spread)

    return ours, rf


def race(train_csv, test_csv) -> dict[str, float]:
    """Score each method on test_csv. Higher is better.

    Keys: "ours" (step17.choose shortlist), "top_overall_rating" (highest OVR),
    "random_search" (uniform sampling), and, when Penalties+ is a goal in both
    files, "penalty_guesser_ours" (step2.guess) and "penalty_guesser_rf"
    (default random forest).
    """
    test_rows, inputs, goals = _load(test_csv)
    if not test_rows or not goals:
        return {}

    train_rows, _, train_goals = _load(train_csv)
    goal_names = [name for name, _ in goals]
    n = len(test_rows)
    budget = min(_BUDGET, n)

    # Shared pool and budget: every tuner-style method samples test_csv builds.
    full = _goal_scores(test_rows, goals)

    def label(i):
        return {name: float(test_rows[i][name]) for name in goal_names}

    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(test_csv, blind)
        out["ours"] = float(full[choose(blind, label, budget=budget)])

    if _OVR in inputs:
        out["top_overall_rating"] = _top_overall_score(full, test_rows, budget)
    out["random_search"] = _random_search_score(full, budget)

    train_goal_names = [name for name, _ in train_goals]
    if (_PEN in goal_names and _PEN in train_goal_names and train_rows):
        p_ours, p_rf = _penalty_scores(train_csv, train_rows, test_rows, inputs)
        out["penalty_guesser_ours"] = p_ours
        out["penalty_guesser_rf"] = p_rf

    return out

