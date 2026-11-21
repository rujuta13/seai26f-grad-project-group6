"""step10.py: race the project's tools against standard rivals on held-out data.

Tuner race: `ours` (step7.choose) vs a random search and a scikit-learn
Gaussian-process expected-improvement tuner, each given the same budget of
measured builds drawn from test_csv.

Guesser race: `guesser_ours` (step2.guess) vs an off-the-shelf
RandomForestRegressor, both predicting every goal of each test build.

All scores are higher-is-better. Tuner scores are the combined z-scored goal
value of the best build found, on the test set. Guesser scores are the negated
mean absolute error per goal, scaled by the test-set spread of that goal.
"""

import csv
import math
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from step1 import _load, _encode
from step2 import guess
from step5 import _goal_scores
from step7 import choose

_BUDGET = 30          # measured builds each tuner may use
_RANDOM_REPEATS = 50  # random-search repeats, averaged for a stable score
_GP_INIT = 5          # initial random design for the GP tuner


def _write_blind(test_csv, path):
    """Copy test_csv to path with every goal cell replaced by '?'."""
    with open(test_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        names = reader.fieldnames or []
        rows = list(reader)
    goal_cols = {n for n in names if n.endswith("+") or n.endswith("-")}
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("?" if k in goal_cols else v) for k, v in r.items()})


def _random_search_score(full, budget):
    """Mean over repeats of the best score found by uniform random sampling."""
    rng = np.random.default_rng(0)
    n = len(full)
    best_found = []
    for _ in range(_RANDOM_REPEATS):
        idx = rng.choice(n, size=budget, replace=False)
        best_found.append(float(full[idx].max()))
    return float(np.mean(best_found))


def _gp_ei_score(test_rows, inputs, full, budget):
    """Gaussian-process expected-improvement tuner over the test builds."""
    n = len(test_rows)
    if inputs:
        X = StandardScaler().fit_transform(_encode(test_rows, inputs))
    else:
        X = np.zeros((n, 1), dtype=float)

    rng = np.random.default_rng(0)
    measured = [int(i) for i in rng.choice(n, size=min(budget, _GP_INIT), replace=False)]

    while len(measured) < budget:
        measured_set = set(measured)
        unl = np.array([i for i in range(n) if i not in measured_set], dtype=int)
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
            normalize_y=True,
            random_state=0,
        )
        gp.fit(X[measured], full[measured])
        mu, sd = gp.predict(X[unl], return_std=True)
        sd = np.maximum(sd, 1e-9)
        best_y = float(full[measured].max())
        gain = mu - best_y
        z = gain / sd
        cdf = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        ei = gain * cdf + sd * pdf
        measured.append(int(unl[int(np.argmax(ei))]))

    return float(full[measured].max())


def _goal_matrix_raw(rows, goals):
    return np.array(
        [[float(r[name]) for name, _ in goals] for r in rows], dtype=float
    )


def _error_score(pred, truth, spread):
    """Negated mean absolute error, per goal scaled by spread, averaged over goals."""
    return -float(np.mean(np.abs(pred - truth) / spread))


def _guesser_scores(train_csv, train_rows, test_rows, inputs, goals):
    """Scores for step2.guess and an off-the-shelf random forest on test_rows."""
    truth = _goal_matrix_raw(test_rows, goals)
    spread = truth.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    # Ours: step2.guess, one query per test build.
    pred_ours = np.zeros_like(truth)
    for i, r in enumerate(test_rows):
        row = {c: r.get(c, "") for c in inputs}
        est = guess(train_csv, row)
        for j, (name, _) in enumerate(goals):
            pred_ours[i, j] = est[name][0]
    ours = _error_score(pred_ours, truth, spread)

    # Off-the-shelf: default random forest on the same encoded inputs.
    if inputs:
        X_all = _encode(train_rows + test_rows, inputs)
        X_tr, X_te = X_all[:len(train_rows)], X_all[len(train_rows):]
    else:
        X_tr = np.zeros((len(train_rows), 1), dtype=float)
        X_te = np.zeros((len(test_rows), 1), dtype=float)
    Y_tr = _goal_matrix_raw(train_rows, goals)
    model = RandomForestRegressor(random_state=0, n_jobs=1)
    model.fit(X_tr, Y_tr)
    pred_rf = np.asarray(model.predict(X_te), dtype=float).reshape(len(test_rows), len(goals))
    rf = _error_score(pred_rf, truth, spread)

    return ours, rf


def race(train_csv, test_csv) -> dict[str, float]:
    """Score each method on test_csv. Higher is better.

    Tuner keys: "ours" (step7.choose), "random_search", "sklearn_gp_ei".
    Guesser keys: "guesser_ours" (step2.guess), "guesser_rf" (default random forest).
    """
    test_rows, inputs, goals = _load(test_csv)
    if not test_rows or not goals:
        return {}

    train_rows, _, _ = _load(train_csv)
    n = len(test_rows)
    budget = min(_BUDGET, n)

    # Tuner race: same budget and the same pool (test_csv builds) for every method.
    full = _goal_scores(test_rows, goals)  # larger is better, z-scored over test set

    def label(i):
        return {name: float(test_rows[i][name]) for name, _ in goals}

    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(test_csv, blind)
        out["ours"] = float(full[choose(blind, label, budget=budget)])

    out["random_search"] = _random_search_score(full, budget)
    out["sklearn_gp_ei"] = _gp_ei_score(test_rows, inputs, full, budget)

    # Guesser race: predict each test build's goals from its inputs.
    if train_rows:
        g_ours, g_rf = _guesser_scores(train_csv, train_rows, test_rows, inputs, goals)
        out["guesser_ours"] = g_ours
        out["guesser_rf"] = g_rf

    return out

