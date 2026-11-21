import csv
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step2 import _columns, _as_float

_N_TREES = 200
_KAPPA = 1.0  # exploration weight on the surrogate's spread


def _read_blind(blind_csv):
    """Read only the input flags; goal cells in a blind file are '?' and are not parsed."""
    inputs, goals = _columns(blind_csv)
    with open(blind_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{c: _as_float(r.get(c, 0.0)) for c in inputs} for r in reader]
    X = np.array([[r[c] for c in inputs] for r in rows], dtype=float)
    X = X.reshape(len(rows), len(inputs))
    return X, inputs, goals


def _standardize_cols(A):
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd


def _scores(meas, k):
    """Scalar score per measured row: mean standardized (signed) goal value.

    meas maps row index -> array of k signed goal values (NaN if missing).
    Rows with any missing value get -inf.
    """
    ids = list(meas)
    V = np.array([meas[i] for i in ids], dtype=float).reshape(len(ids), k)
    good = np.all(np.isfinite(V), axis=1)
    out = {i: -math.inf for i in ids}
    if good.any():
        Z = _standardize_cols_with_stats(V[good])
        s = Z.mean(axis=1)
        for i, v in zip([ids[j] for j in np.flatnonzero(good)], s):
            out[i] = float(v)
    return out


def _standardize_cols_with_stats(G):
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    return (G - mu) / sd


def _farthest_first(Z, k, rng):
    """Pick k well-spread rows of Z (greedy farthest-point sampling)."""
    n = len(Z)
    first = int(rng.integers(n))
    chosen = [first]
    mind = np.linalg.norm(Z - Z[first], axis=1)
    mind[first] = -1.0
    while len(chosen) < k:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(Z - Z[nxt], axis=1))
        mind[chosen] = -1.0
    return chosen


def _farthest_from(Z, measured_ids, candidates):
    """Candidate row farthest (in standardized flag space) from all measured rows."""
    D = np.linalg.norm(
        Z[candidates][:, None, :] - Z[measured_ids][None, :, :], axis=2
    ).min(axis=1)
    return int(candidates[int(np.argmax(D))])


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Find a good flag set by benchmarking at most `budget` builds.

    label(i) returns the goals of row i as a dict (at most `budget` calls).
    If `goal` names one goal, only that goal is optimised; otherwise all goals
    are combined as the mean of their standardized, sign-adjusted values.
    Returns the index of the best measured row.
    """
    X, inputs, goals = _read_blind(blind_csv)
    if not goals:
        raise ValueError("blind_csv has no goal columns")
    n = len(X)
    if n == 0:
        raise ValueError("blind_csv has no rows")
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")

    if goal is None:
        use = list(range(len(goals)))
    else:
        if goal not in goals:
            raise ValueError(f"unknown goal {goal!r}; goals are {goals}")
        use = [goals.index(goal)]
    use_names = [goals[j] for j in use]
    k = len(use)
    signs = np.array([1.0 if g.endswith("+") else -1.0 for g in use_names])

    spent = 0
    meas = {}

    def measure(i):
        nonlocal spent
        if spent >= budget:
            raise RuntimeError("benchmark budget exceeded")
        spent += 1
        d = label(i)
        vals = np.full(k, np.nan)
        for j, g in enumerate(use_names):
            try:
                vals[j] = float(d[g]) * signs[j]
            except (KeyError, TypeError, ValueError):
                vals[j] = np.nan
        meas[i] = vals

    rng = np.random.default_rng(0)

    if X.shape[1] == 0:
        # No flags to model: nothing to learn from, just measure in file order.
        for i in range(min(n, budget)):
            measure(i)
    else:
        Z = _standardize_cols(X)
        n_init = min(n, budget, max(3, budget // 3))
        for i in _farthest_first(Z, n_init, rng):
            measure(int(i))

        while spent < budget and len(meas) < n:
            sc = _scores(meas, k)
            ok_ids = [i for i in meas if math.isfinite(sc[i])]
            unmeasured = np.array([i for i in range(n) if i not in meas], dtype=int)

            if len(ok_ids) < 2:
                pick = _farthest_from(Z, list(meas), unmeasured)
            else:
                y = np.array([sc[i] for i in ok_ids], dtype=float)
                model = RandomForestRegressor(
                    n_estimators=_N_TREES, min_samples_leaf=1,
                    random_state=spent, n_jobs=1,
                )
                model.fit(X[ok_ids], y)
                preds = np.array([t.predict(X[unmeasured]) for t in model.estimators_])
                acq = preds.mean(axis=0) + _KAPPA * preds.std(axis=0)
                pick = int(unmeasured[int(np.argmax(acq))])
            measure(pick)

    sc = _scores(meas, k)
    best_i = max(meas, key=lambda i: (sc[i], -i))
    return int(best_i)

