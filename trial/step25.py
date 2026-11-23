import numpy as np

from step1 import _load
from step2 import _columns
from step4 import _goal_layout
from step15 import _blame_for, _IDENTIFIERS

# The client's two questions: customers who never respond to campaigns
# (weak on RESPONSE+) and customers who complain (high COMPLAIN-, which is
# negated in G, so fewer complaints means a higher, better value).
_FOCUS = ("RESPONSE+", "COMPLAIN-")


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Bad means never responding to campaigns (weak RESPONSE+) or complaining
    (weak COMPLAIN-). Each focus goal gets its own blame ranking from
    step15._blame_for, and the rankings are averaged, so inputs that explain
    either problem rise to the top. Identifier columns are never blamed and
    sink to the end.
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
    if focus:
        badness_list = [-Z[:, g] for g in focus]
    else:
        # Neither focus goal exists: fall back to all goals together.
        badness_list = [-Z.mean(axis=1)]

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

