import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from step1 import _load
from step2 import _columns, _as_float
from step6 import _standardize
from step7 import choose
from step10 import _BUDGET, _GUESS_ROWS, _composite, _pick, _r2
from step22 import guess

_INCOME = "INCOME"
_RESPONSE = "RESPONSE+"


def race(train_csv, test_csv) -> dict[str, float]:
    """Race our targeting and response guesser against simple rivals on test_csv.

    Targeting scores: the true mean standardized goal of the customer each
    method picks after measuring at most _BUDGET test rows. Rivals are a
    uniform random sample and the highest-income customers (the client's
    "simply mail our highest-income customers" plan). Higher is better.

    Guesser scores: R^2 of predicted vs true RESPONSE+ on a subsample of test
    rows, for step22.guess ("guess_ours") and a gradient boosting model
    ("guess_off_the_shelf"). Higher is better. Compare within each group.
    """
    inputs, test_goals = _columns(test_csv)
    _, train_goals = _columns(train_csv)
    dt = _load(test_csv)
    Gt, Xt, rows = dt["G"], dt["X"], dt["rows"]
    nt = len(Gt)
    zero = {"ours": 0.0, "random": 0.0, "top_income": 0.0,
            "guess_ours": 0.0, "guess_off_the_shelf": 0.0}
    if not test_goals or nt == 0:
        return zero

    _, mu, sd, _ = _standardize(train_csv)
    true = _composite(Gt, mu, sd)
    k = Gt.shape[1]
    budget = min(_BUDGET, nt)
    result = {}

    # Our targeting: the step7 tuner, measuring through the label callback.
    ours_idx = choose(test_csv, lambda i: rows[i], budget=budget)
    result["ours"] = float(true[ours_idx])

    # Rival 1: uniform random sample of the same size.
    rng = np.random.default_rng(0)
    rand_pick = rng.choice(nt, size=budget, replace=False)
    meas = {int(i): Gt[i].copy() for i in rand_pick}
    result["random"] = float(true[_pick(meas, k)])

    # Rival 2: mail the highest-income customers (highest INCOME first).
    income = np.array([_as_float(r.get(_INCOME, 0.0)) for r in rows], dtype=float)
    order = sorted(range(nt), key=lambda i: (-income[i], i))
    meas = {int(i): Gt[i].copy() for i in order[:budget]}
    result["top_income"] = float(true[_pick(meas, k)])

    # Response guesser versus an off-the-shelf model, on RESPONSE+ itself.
    if inputs and _RESPONSE in test_goals and _RESPONSE in train_goals:
        dtr = _load(train_csv)
        Xtr, train_rows = dtr["X"], dtr["rows"]
        n_g = min(_GUESS_ROWS, nt)
        sub = np.random.default_rng(2).choice(nt, size=n_g, replace=False)
        truth = np.array([float(rows[int(i)][_RESPONSE]) for i in sub], dtype=float)

        pred_ours = []
        for i in sub:
            est = guess(train_csv, rows[int(i)])
            if _RESPONSE in est:
                pred_ours.append(float(est[_RESPONSE][0]))
            else:
                pred_ours.append(0.0)
        result["guess_ours"] = _r2(truth, pred_ours)

        ytr = np.array([float(r[_RESPONSE]) for r in train_rows], dtype=float)
        off = GradientBoostingRegressor(random_state=0)
        off.fit(Xtr, ytr)
        result["guess_off_the_shelf"] = _r2(truth, off.predict(Xt[sub]))
    else:
        result["guess_ours"] = 0.0
        result["guess_off_the_shelf"] = 0.0

    return {name: float(v) for name, v in result.items()}

