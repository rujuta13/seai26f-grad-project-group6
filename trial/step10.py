"""step10.py: race the tuner and the guesser against standard rivals on test_csv.

Tuner rivals (each may measure at most `budget` test builds, then picks one):
  ours       step7.choose (phased exploration and exploitation)
  random     uniform random measurements, same Pareto-then-mean-rank pick rule
  smbo_rf    sequential model-based search: a random-forest surrogate on the
             standardised sum of goals proposes the next build to measure

Guesser rivals (predict every goal of each test build from its inputs):
  ours_guess step2.guess (k-nearest-neighbour median)
  rf_guess   random forest regressor on standardised goals

Scores (lower is better for every method):
  tuner:    mean-goal-rank position of the chosen build among all test builds,
            scaled to [0, 1]; 0 means the best test build on average across goals.
  guesser:  mean absolute error per goal divided by the training goal spread,
            averaged over goals.
"""
from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _classify, _goal_matrix, _load
from step2 import guess
from step4 import _encode_row, _encode_train
from step6 import _mean_rank
from step7 import choose

_BUDGET = 30        # measurements allowed to every tuner
_SEED = 0           # fixed seed so the race is repeatable
_SMBO_INIT = 5      # random builds measured before the surrogate takes over
_RF_TREES_TUNE = 100
_RF_TREES_GUESS = 200


def _write_blind(header, rows, goal_cols, path):
    """Copy the test file with every goal cell set to '?'."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            cells = list(r)
            for c in goal_cols:
                cells[c] = "?"
            w.writerow(cells)


def _tuner_score(G, chosen):
    """Position of the chosen build by mean goal rank among all test builds, in [0, 1]."""
    n = G.shape[0]
    if n < 2 or G.shape[1] == 0:
        return 0.0
    mr = _mean_rank(G)
    return float(mr[int(chosen)]) / (n - 1)


def _pick_measured(G, ids):
    """Best measured build: Pareto front of the measured builds, then lowest mean rank."""
    ids = np.asarray(sorted({int(i) for i in ids}), dtype=int)
    sub = G[ids]
    le = (sub[None, :, :] <= sub[:, None, :]).all(axis=2)
    lt = (sub[None, :, :] < sub[:, None, :]).any(axis=2)
    front = np.flatnonzero(~(le & lt).any(axis=1))
    score = _mean_rank(sub[front])
    return int(ids[front[int(np.argmin(score))]])


def _smbo(G, F, budget, seed):
    """Sequential model-based search with a random-forest surrogate; returns measured ids."""
    n = G.shape[0]
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    y = ((G - mu) / sd).sum(axis=1)

    rng = np.random.default_rng(seed)
    init = min(budget, _SMBO_INIT)
    measured = [int(i) for i in rng.choice(n, size=init, replace=False)]
    while len(measured) < budget:
        model = RandomForestRegressor(n_estimators=_RF_TREES_TUNE, random_state=seed)
        model.fit(F[measured], y[measured])
        rest = np.setdiff1d(np.arange(n), measured)
        pred = model.predict(F[rest])
        measured.append(int(rest[int(np.argmin(pred))]))
    return measured


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores of our tuner and guesser against standard rivals on test_csv.

    Keys: "ours", "random", "smbo_rf" (tuners, see module docstring) and
    "ours_guess", "rf_guess" (goal predictors). All scores are lower-is-better.
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
    goal_names = [header[c] for c in goal_cols]

    n = len(test_rows)
    budget = min(_BUDGET, n)
    G = _goal_matrix(test_rows, maximise, minimise)
    truth = np.array([[float(r[c]) for c in goal_cols] for r in test_rows], dtype=float)

    # Encoded test inputs, using the category codes learned from the training data.
    _, specs = _encode_train(train_rows, inputs)
    test_dicts = [dict(zip(header, r)) for r in test_rows]
    if inputs:
        F = np.array([_encode_row(d, header, inputs, specs) for d in test_dicts], dtype=float)
    else:
        F = np.zeros((n, 1), dtype=float)

    scores: dict[str, float] = {}

    # Tuner rivals.
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(header, test_rows, goal_cols, blind)

        def label(i):
            return {header[c]: float(test_rows[i][c]) for c in goal_cols}

        scores["ours"] = _tuner_score(G, choose(blind, label, budget=budget))

    rng = np.random.default_rng(_SEED)
    rand_ids = rng.choice(n, size=budget, replace=False)
    scores["random"] = _tuner_score(G, _pick_measured(G, rand_ids))

    smbo_ids = _smbo(G, F, budget, _SEED)
    scores["smbo_rf"] = _tuner_score(G, _pick_measured(G, smbo_ids))

    # Goal predictors.
    Ytrain = np.array([[float(r[c]) for c in goal_cols] for r in train_rows], dtype=float)
    mu_t = Ytrain.mean(axis=0)
    scale = Ytrain.std(axis=0)
    scale[scale == 0] = 1.0

    def _err(pred):
        return float(np.mean(np.abs(pred - truth) / scale))

    # Our guesser: step2.guess on the input columns only.
    ours_pred = np.empty_like(truth)
    for i, row in enumerate(test_rows):
        inputs_only = {header[c]: row[c] for c in inputs}
        est = guess(train_csv, inputs_only)
        ours_pred[i] = [est[name][0] for name in goal_names]
    scores["ours_guess"] = _err(ours_pred)

    # Off-the-shelf rival: random forest on the same encoded inputs.
    if inputs:
        Xtrain, _ = _encode_train(train_rows, inputs)
    else:
        Xtrain = np.zeros((len(train_rows), 1), dtype=float)
    Ys = (Ytrain - mu_t) / scale
    forest = RandomForestRegressor(n_estimators=_RF_TREES_GUESS, random_state=_SEED)
    forest.fit(Xtrain, Ys)
    rf_pred = forest.predict(F) * scale + mu_t
    scores["rf_guess"] = _err(rf_pred)

    return scores

