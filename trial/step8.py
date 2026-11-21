import itertools

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _load
from step2 import _columns, _as_float
from step6 import _standardize

_MAX_CHANGES = 3          # search up to this many changed flags
_MAX_CANDIDATES = 50_000  # cap on candidate flag sets evaluated per change count


def fix(train_csv, row, frozen=()) -> dict:
    """Smallest set of input changes that makes a bad row good.

    "Good" is the same verdict step6.why uses: the average standardized goal
    score is at least the training median. The goals of `row` are unknown, so
    each candidate flag set is scored with a random forest fitted on the
    training rows. Columns named in `frozen` are never changed. Returns the
    row's inputs with the fewest changes found (fewest changed flags first,
    then highest predicted score).
    """
    inputs, goals = _columns(train_csv)
    base = np.array([_as_float(row.get(c, 0.0)) for c in inputs], dtype=float)

    def out(v):
        return {c: float(x) for c, x in zip(inputs, v)}

    if not goals or not inputs:
        return out(base)

    d = _load(train_csv)
    X = d["X"]
    if len(X) < 4:
        return out(base)

    G, mu, sd, Z = _standardize(train_csv)
    median = float(np.median(Z.mean(axis=1)))

    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
    )
    model.fit(X, G)

    def scores(C):
        P = model.predict(C).reshape(len(C), G.shape[1])
        return ((P - mu) / sd).mean(axis=1)

    frozen_set = set(frozen)
    free = [j for j, c in enumerate(inputs) if c not in frozen_set]

    s0 = float(scores(base.reshape(1, -1))[0])
    if s0 >= median or not free:
        return out(base)

    best_v, best_s = base, s0
    alts = {
        j: [float(v) for v in np.unique(X[:, j]) if v != base[j]]
        for j in free
    }

    for k in range(1, min(_MAX_CHANGES, len(free)) + 1):
        cands = []
        for cols in itertools.combinations(free, k):
            if any(not alts[j] for j in cols):
                continue
            for vals in itertools.product(*(alts[j] for j in cols)):
                v = base.copy()
                for j, x in zip(cols, vals):
                    v[j] = x
                cands.append(v)
                if len(cands) >= _MAX_CANDIDATES:
                    break
            if len(cands) >= _MAX_CANDIDATES:
                break
        if not cands:
            continue

        C = np.array(cands, dtype=float)
        s = scores(C)
        good = np.flatnonzero(s >= median)
        if good.size:
            i = int(good[np.argmax(s[good])])
            return out(C[i])

        i = int(np.argmax(s))
        if s[i] > best_s:
            best_v, best_s = C[i], float(s[i])

    return out(best_v)

