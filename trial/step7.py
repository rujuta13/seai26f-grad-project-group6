"""step7.py: choose the best flag set while benchmarking at most `budget` builds.

Active search: a space-filling initial design, then repeated random-forest
upper-confidence-bound picks over the unmeasured builds. Only the goals being
optimised are used, so `goal` switches the search to a single goal.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from step1 import _load, _encode
from step5 import _goal_scores

_UCB_KAPPA = 1.0
_N_TREES = 200


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, calling label(i) at most `budget` times.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict. If `goal` names one goal, only that goal is optimised.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")

    rows, inputs, goals = _load(blind_csv)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no rows")
    if not goals:
        raise ValueError("blind_csv has no goal columns")

    if goal is None:
        use = list(goals)
    else:
        use = [(g, s) for g, s in goals if g == goal]
        if not use:
            raise ValueError(f"unknown goal {goal!r}")
    names = [g for g, _ in use]

    # Feature matrix over the inputs (goal cells are blind, so only inputs are used).
    if inputs:
        X = StandardScaler().fit_transform(_encode(rows, inputs))
    else:
        X = np.zeros((n, 1), dtype=float)

    cap = min(budget, n)
    measured: dict[int, dict[str, float]] = {}

    def measure(i):
        d = label(i)
        measured[i] = {name: float(d[name]) for name in names}

    def scores(idx):
        # Combined z-scored goal score over the measured rows, larger is better.
        return _goal_scores([measured[i] for i in idx], use)

    # Initial space-filling design: farthest-point sampling on the inputs.
    n_init = min(cap, max(3, cap // 3))
    chosen = [0]
    dist = np.linalg.norm(X - X[0], axis=1)
    while len(chosen) < n_init:
        masked = dist.copy()
        masked[chosen] = -np.inf
        nxt = int(np.argmax(masked))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    for i in chosen:
        measure(i)

    # Sequential search: model the scores, measure the most promising unmeasured build.
    while len(measured) < cap:
        lab = sorted(measured)
        unl = np.array([i for i in range(n) if i not in measured], dtype=int)
        if len(lab) < 5:
            # Too few points to model: measure the build farthest from all measured ones.
            d = np.min(
                np.linalg.norm(X[unl][:, None, :] - X[lab][None, :, :], axis=2),
                axis=1,
            )
            pick = int(unl[np.argmax(d)])
        else:
            y = scores(lab)
            model = RandomForestRegressor(
                n_estimators=_N_TREES,
                random_state=0,
                n_jobs=1,
            )
            model.fit(X[lab], y)
            preds = np.stack([t.predict(X[unl]) for t in model.estimators_])
            acq = preds.mean(axis=0) + _UCB_KAPPA * preds.std(axis=0)
            pick = int(unl[np.argmax(acq)])
        measure(pick)

    # Best measured build on the goals being optimised.
    lab = sorted(measured)
    y = scores(lab)
    return int(lab[int(np.argmax(y))])

