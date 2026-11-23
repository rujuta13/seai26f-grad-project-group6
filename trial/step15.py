import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _load
from step2 import _columns
from step4 import _goal_layout

# Goals the client asked about: weakness from the spot, and physical strength.
_FOCUS = ("Penalties+", "Strength+")
# Columns that identify a record rather than describe it; never used as causes.
_IDENTIFIERS = {"num", "Rank", "Name", "url"}


def _blame_for(X, badness):
    """Blame per input column for one badness vector (higher = worse).

    Uses the same rule as step5.blame: an input is blamed only if its "on"
    state (any value above the column minimum) goes with worse rows.
    """
    m = X.shape[1]
    out = np.zeros(m)
    if m == 0 or np.all(badness == badness[0]):
        return out

    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
    )
    model.fit(X, badness)
    imp = model.feature_importances_

    for j in range(m):
        col = X[:, j]
        on = col != col.min()
        off = ~on
        if on.any() and off.any() and imp[j] > 0:
            if badness[on].mean() > badness[off].mean():
                out[j] = imp[j]
    return out


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Bad means weak on Penalties+ (weak from the spot) or weak on Strength+
    (physically weak). Each focus goal gets its own blame ranking, and the two
    rankings are averaged, so inputs that explain either weakness rise to the top.
    Identifier columns are never blamed and sink to the end.
    """
    inputs, _ = _columns(train_csv)
    if not inputs:
        return []

    d = _load(train_csv)
    X = d["X"]
    G = d["G"]
    n = len(X)
    if n < 2 or G.shape[1] == 0:
        return list(inputs)

    maxg, ming = _goal_layout(train_csv)
    names = maxg + ming  # same column order as step1._load's G

    # Standardize goals so higher is better on every column.
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (G - mu) / sd

    focus = [names.index(g) for g in _FOCUS if g in names]
    if not focus:
        # None of the focus goals exist: fall back to all goals together.
        focus = list(range(G.shape[1]))
        badness_list = [-Z.mean(axis=1)]
    else:
        badness_list = [-Z[:, g] for g in focus]

    feat = [j for j, c in enumerate(inputs) if c not in _IDENTIFIERS]
    if not feat:
        feat = list(range(len(inputs)))
    Xf = X[:, feat]

    total = np.zeros(len(inputs))
    for badness in badness_list:
        b = _blame_for(Xf, badness)
        total[feat] += b / len(badness_list)

    order = sorted(range(len(inputs)), key=lambda j: (-total[j], j))
    return [inputs[j] for j in order]

