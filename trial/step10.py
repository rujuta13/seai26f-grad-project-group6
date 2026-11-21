import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from step1 import _load
from step2 import _columns, guess
from step4 import _goal_layout, _goal_vec
from step6 import _standardize
from step7 import choose, _scores

_BUDGET = 30        # benchmark builds each tuner may measure
_GUESS_ROWS = 25    # test rows used to compare the guessers (guess() is slow)


def _composite(G, mu, sd):
    """Mean standardized goal score per row, using training mean and sd."""
    return ((G - mu) / sd).mean(axis=1)


def _r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst == 0.0:
        return 0.0
    return 1.0 - float(np.sum((y - p) ** 2)) / sst


def _pick(meas, k):
    """Choose the best measured row with the same rule step7.choose uses."""
    sc = _scores(meas, k)
    return max(meas, key=lambda i: (sc[i], -i))


def race(train_csv, test_csv) -> dict[str, float]:
    """Race the tuner (step7.choose) against random search and an off-the-shelf
    surrogate tuner on test_csv, and compare the guesser (step2.guess) with an
    off-the-shelf regressor.

    Tuner scores: the true mean standardized goal of the build each method
    picks, after measuring at most _BUDGET test builds. Higher is better.
    Guesser scores: R^2 of predicted vs true composite score on a subsample of
    test rows. Higher is better. Compare tuners with tuners and guessers with
    guessers.
    """
    _, goals = _columns(test_csv)
    inputs, _ = _columns(test_csv)
    dt = _load(test_csv)
    Gt, Xt, rows = dt["G"], dt["X"], dt["rows"]
    nt = len(Gt)
    if not goals or nt == 0:
        return {"ours": 0.0, "random": 0.0, "off_the_shelf": 0.0,
                "guess_ours": 0.0, "guess_off_the_shelf": 0.0}

    _, mu, sd, _ = _standardize(train_csv)
    true = _composite(Gt, mu, sd)
    k = Gt.shape[1]
    budget = min(_BUDGET, nt)

    def label(i):
        return rows[i]

    result = {}

    # Ours: the step7 tuner, measuring through the label callback.
    ours_idx = choose(test_csv, label, budget=budget)
    result["ours"] = float(true[ours_idx])

    # Random search: measure a uniform random sample of the same size.
    rng = np.random.default_rng(0)
    rand_pick = rng.choice(nt, size=budget, replace=False)
    meas = {int(i): Gt[i].copy() for i in rand_pick}
    result["random"] = float(true[_pick(meas, k)])

    # Off-the-shelf surrogate tuner: sklearn gradient boosting, greedy picks.
    if Xt.shape[1] == 0:
        meas = {int(i): Gt[i].copy() for i in range(budget)}
        result["off_the_shelf"] = float(true[_pick(meas, k)])
    else:
        init = min(3, budget)
        measured = [int(i) for i in rng.choice(nt, size=init, replace=False)]
        while len(measured) < budget:
            taken = set(measured)
            unm = np.array([i for i in range(nt) if i not in taken], dtype=int)
            model = GradientBoostingRegressor(random_state=0)
            model.fit(Xt[measured], true[measured])
            measured.append(int(unm[int(np.argmax(model.predict(Xt[unm])))]))
        meas = {i: Gt[i].copy() for i in measured}
        result["off_the_shelf"] = float(true[_pick(meas, k)])

    # Guesser comparison on a subsample of test rows.
    if inputs and goals:
        dtr = _load(train_csv)
        Xtr, Gtr = dtr["X"], dtr["G"]
        n_g = min(_GUESS_ROWS, nt)
        sub = np.random.default_rng(2).choice(nt, size=n_g, replace=False)
        truth = true[sub]

        maxg, ming = _goal_layout(train_csv)
        pred_ours = []
        for i in sub:
            est = guess(train_csv, rows[int(i)])
            mids = {g: v[0] for g, v in est.items()}
            vec = np.array(_goal_vec(mids, maxg, ming), dtype=float)
            pred_ours.append(float(((vec - mu) / sd).mean()))
        result["guess_ours"] = _r2(truth, pred_ours)

        ytr = _composite(Gtr, mu, sd)
        off = GradientBoostingRegressor(random_state=0)
        off.fit(Xtr, ytr)
        result["guess_off_the_shelf"] = _r2(truth, off.predict(Xt[sub]))
    else:
        result["guess_ours"] = 0.0
        result["guess_off_the_shelf"] = 0.0

    return {k_: float(v) for k_, v in result.items()}

