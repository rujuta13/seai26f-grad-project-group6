import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _load
from step2 import _columns


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    inputs, _ = _columns(train_csv)
    if not inputs:
        return []

    d = _load(train_csv)
    X = d["X"]
    G = d["G"]
    n = len(X)
    if n < 2 or G.shape[1] == 0:
        return list(inputs)

    # Same standardization as step1.best: higher score is better on every goal.
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    score = ((G - mu) / sd).mean(axis=1)
    badness = -score  # higher means worse

    if np.all(badness == badness[0]):
        return list(inputs)

    # Importance of each input for predicting badness.
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
    )
    model.fit(X, badness)
    imp = model.feature_importances_

    # An input is only blamed if its "on" state (any value above the column
    # minimum) goes with worse rows than its "off" state. Inputs whose
    # presence helps get zero blame and sink to the end.
    blame_scores = np.zeros(len(inputs))
    for j in range(len(inputs)):
        col = X[:, j]
        on = col != col.min()
        off = ~on
        if on.any() and off.any() and imp[j] > 0:
            if badness[on].mean() > badness[off].mean():
                blame_scores[j] = imp[j]

    order = sorted(range(len(inputs)), key=lambda j: (-blame_scores[j], j))
    return [inputs[j] for j in order]

