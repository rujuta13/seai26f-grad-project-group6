import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from step1 import _load
from step2 import _columns, _as_float, guess
from step4 import _goal_layout
from step6 import _standardize
from step7 import choose
from step10 import _BUDGET, _GUESS_ROWS, _composite, _pick, _r2

_PENALTY = "Penalties+"
_OVR = "OVR"


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare the shortlist tuner with simple rivals on test_csv, and the
    penalty guesser with an off-the-shelf model.

    Shortlist scores: the true mean standardized goal of the build each method
    picks after measuring at most _BUDGET test builds. Rivals are a uniform
    random list and a plain "top overall rating" list (highest OVR first).
    Penalty guesser scores: R^2 of predicted vs true Penalties+ on a subsample
    of test rows, for step2.guess ("guess_ours") and for a gradient boosting
    model ("guess_off_the_shelf"). Higher is better. Compare within each group.
    """
    inputs, goals = _columns(test_csv)
    dt = _load(test_csv)
    Gt, Xt, rows = dt["G"], dt["X"], dt["rows"]
    nt = len(Gt)
    zero = {"ours": 0.0, "random": 0.0, "top_overall": 0.0,
            "guess_ours": 0.0, "guess_off_the_shelf": 0.0}
    if not goals or nt == 0:
        return zero

    _, mu, sd, _ = _standardize(train_csv)
    true = _composite(Gt, mu, sd)
    k = Gt.shape[1]
    budget = min(_BUDGET, nt)
    result = {}

    # Shortlist: the step7 tuner, measuring through the label callback.
    ours_idx = choose(test_csv, lambda i: rows[i], budget=budget)
    result["ours"] = float(true[ours_idx])

    # Rival 1: uniform random list of the same size.
    rng = np.random.default_rng(0)
    rand_pick = rng.choice(nt, size=budget, replace=False)
    meas = {int(i): Gt[i].copy() for i in rand_pick}
    result["random"] = float(true[_pick(meas, k)])

    # Rival 2: the simple top-overall-rating list (highest OVR first).
    ovr = np.array([_as_float(r.get(_OVR, 0.0)) for r in rows], dtype=float)
    order = sorted(range(nt), key=lambda i: (-ovr[i], i))
    meas = {int(i): Gt[i].copy() for i in order[:budget]}
    result["top_overall"] = float(true[_pick(meas, k)])

    # Penalty guesser versus an off-the-shelf model, on Penalties+ itself.
    _, train_goals = _columns(train_csv)
    if inputs and _PENALTY in goals and _PENALTY in train_goals:
        dtr = _load(train_csv)
        Xtr, train_rows = dtr["X"], dtr["rows"]
        n_g = min(_GUESS_ROWS, nt)
        sub = np.random.default_rng(2).choice(nt, size=n_g, replace=False)
        truth = np.array([float(rows[int(i)][_PENALTY]) for i in sub], dtype=float)

        pred_ours = []
        for i in sub:
            est = guess(train_csv, rows[int(i)])
            pred_ours.append(float(est[_PENALTY][0]))
        result["guess_ours"] = _r2(truth, pred_ours)

        ytr = np.array([float(r[_PENALTY]) for r in train_rows], dtype=float)
        off = GradientBoostingRegressor(random_state=0)
        off.fit(Xtr, ytr)
        result["guess_off_the_shelf"] = _r2(truth, off.predict(Xt[sub]))
    else:
        result["guess_ours"] = 0.0
        result["guess_off_the_shelf"] = 0.0

    return {name: float(v) for name, v in result.items()}

