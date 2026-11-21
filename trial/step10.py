import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score

from step1 import _read, _columns, _to_float_column
from step2 import _encode_inputs, guess
from step7 import choose

_BUDGET = 30
_TRIALS = 20
_KAPPA = 1.0


def _oriented(rows, goals):
    """Oriented goal matrix (larger is better), NaN where a value is unusable."""
    return np.column_stack(
        [sign * _to_float_column(rows, name) for name, sign in goals]
    )


def _norm_bounds(S):
    """Per-goal min and span over the finite values of S."""
    G = S.shape[1]
    lo = np.zeros(G)
    hi = np.ones(G)
    for g in range(G):
        col = S[:, g]
        fin = col[np.isfinite(col)]
        if fin.size:
            lo[g] = float(fin.min())
            hi[g] = float(fin.max())
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return lo, span


def _quality_rows(S, lo, span):
    """Mean min-max normalised goal value per row; unusable cells count as 0."""
    Z = (S - lo) / span
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z.mean(axis=1)


def _select_from(idxs, S):
    """Labelled-set selection rule: best normalised goal mean among labelled rows."""
    idxs = np.asarray(idxs, dtype=int)
    M = S[idxs]
    ok = np.all(np.isfinite(M), axis=1)
    if not np.any(ok):
        return int(idxs[0])
    Mo = M[ok]
    io = idxs[ok]
    lo, span = _norm_bounds(Mo)
    score = _quality_rows(Mo, lo, span)
    return int(io[int(np.argmax(score))])


def _objective(M):
    """Scalar objective per labelled row: sum of z-scored oriented goals."""
    obj = np.full(M.shape[0], np.nan)
    fin_rows = np.all(np.isfinite(M), axis=1)
    if not np.any(fin_rows):
        return obj
    Mf = M[fin_rows]
    mu = Mf.mean(axis=0)
    sd = Mf.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    obj[fin_rows] = ((Mf - mu) / sd).sum(axis=1)
    return obj


def _features(rows, inputs):
    """Standardised encoded inputs for every row."""
    n = len(rows)
    if inputs:
        X, _ = _encode_inputs(rows, inputs)
    else:
        X = np.zeros((n, 1))
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _gp_pick(Z, S, B, rng):
    """Off-the-shelf rival tuner: GP surrogate with an upper-confidence-bound rule."""
    n = S.shape[0]
    first_k = min(B, max(2, B // 3))
    labelled = [int(i) for i in rng.choice(n, size=first_k, replace=False)]
    while len(labelled) < B:
        idx = np.array(labelled, dtype=int)
        unl = np.setdiff1d(np.arange(n), idx)
        if unl.size == 0:
            break
        obj = _objective(S[idx])
        ok = np.isfinite(obj)
        if ok.sum() < 3:
            nxt = int(rng.choice(unl))
        else:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=0
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(Z[idx[ok]], obj[ok])
                mu, sd = gp.predict(Z[unl], return_std=True)
            nxt = int(unl[int(np.argmax(mu + _KAPPA * sd))])
        labelled.append(nxt)
    return _select_from(np.array(labelled, dtype=int), S)


def _mean_r2(test_rows, goals, preds):
    """Mean per-goal R^2 of predictions against finite test goal values."""
    scores = []
    for name, _sign in goals:
        y = _to_float_column(test_rows, name)
        p = preds[name]
        mask = np.isfinite(y) & np.isfinite(p)
        if mask.sum() >= 2 and np.var(y[mask]) > 0:
            scores.append(float(r2_score(y[mask], p[mask])))
    return float(np.mean(scores)) if scores else 0.0


def _guesser_r2(train_csv, test_rows, goals):
    preds = {name: np.full(len(test_rows), np.nan) for name, _ in goals}
    for i, row in enumerate(test_rows):
        est = guess(train_csv, row)
        for name, _ in goals:
            preds[name][i] = est[name][0]
    return _mean_r2(test_rows, goals, preds)


def _forest_r2(train_csv, test_rows, goals):
    fieldnames, train_rows = _read(train_csv)
    inputs, _ = _columns(fieldnames)
    m = len(test_rows)
    n = len(train_rows)
    if n == 0:
        return 0.0
    if inputs:
        X, encode_row = _encode_inputs(train_rows, inputs)
        Xte = np.vstack([encode_row(r) for r in test_rows]).astype(float)
    else:
        X = np.zeros((n, 1))
        Xte = np.zeros((m, 1))
    preds = {}
    for name, _sign in goals:
        y = _to_float_column(train_rows, name)
        fin = np.isfinite(y)
        p = np.full(m, np.nan)
        if fin.sum() >= 2:
            model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
            model.fit(X[fin], y[fin])
            p = model.predict(Xte)
        preds[name] = p
    return _mean_r2(test_rows, goals, preds)


def race(train_csv, test_csv) -> dict[str, float]:
    """Race the tuner against rivals on test_csv, and the guesser against a
    standard model. Higher scores are better.

    Tuner keys (score = mean min-max normalised goal value of the chosen build,
    in [0, 1], with a budget of 30 labelled rows):
      - "ours": step7.choose
      - "random_search": mean over repeated random draws of 30 rows
      - "gp_tuner": scikit-learn Gaussian-process UCB tuner
    Guesser keys (score = mean per-goal R^2 of predicted goal medians):
      - "guess:ours": step2.guess
      - "guess:random_forest": scikit-learn RandomForestRegressor
    """
    fieldnames, test_rows = _read(test_csv)
    inputs, goals = _columns(fieldnames)
    n = len(test_rows)
    keys = ["ours", "random_search", "gp_tuner", "guess:ours", "guess:random_forest"]
    if n == 0 or not goals:
        return {k: 0.0 for k in keys}

    names = [g for g, _ in goals]
    S = _oriented(test_rows, goals)
    lo, span = _norm_bounds(S)
    qual = _quality_rows(S, lo, span)
    B = min(_BUDGET, n)

    def label(i):
        return {g: test_rows[i].get(g) for g in names}

    scores: dict[str, float] = {}

    # Our tuner.
    ours_idx = choose(test_csv, label, budget=B)
    scores["ours"] = float(qual[ours_idx])

    # Random search rival, averaged over repeated draws.
    rs = []
    for t in range(_TRIALS):
        rng = np.random.default_rng(t)
        pick = rng.choice(n, size=B, replace=False)
        rs.append(float(qual[_select_from(pick, S)]))
    scores["random_search"] = float(np.mean(rs))

    # Standard library-style tuner rival (GP with UCB).
    Z = _features(test_rows, inputs)
    gp_rng = np.random.default_rng(0)
    gp_idx = _gp_pick(Z, S, B, gp_rng)
    scores["gp_tuner"] = float(qual[gp_idx])

    # Guesser versus off-the-shelf model.
    scores["guess:ours"] = _guesser_r2(train_csv, test_rows, goals)
    scores["guess:random_forest"] = _forest_r2(train_csv, test_rows, goals)

    return scores

