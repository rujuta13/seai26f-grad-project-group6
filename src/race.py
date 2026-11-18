"""Rivals built from scikit-learn, to race our own tuner and guesser against."""
from sklearn.ensemble import RandomForestRegressor


def bits(t, rows):
    return [[float(r.cells[c.at] == "1") for c in t.x] for r in rows]


def forest_pick(t, rng, budget=30, score=None, start=6, pool=100):
    """The `choose.pick` loop with a random forest as the guesser, the usual off-the-shelf tuner."""
    score = score or t.disty
    seen = {id(r): (r, score(r)) for r in rng.sample(t.rows, start)}
    while len(seen) < budget:
        rows = [r for r, _ in seen.values()]
        model = RandomForestRegressor(n_estimators=30, random_state=rng.randrange(10**6))
        model.fit(bits(t, rows), [s for _, s in seen.values()])
        cands = [r for r in rng.sample(t.rows, pool) if id(r) not in seen]
        best = min(zip(model.predict(bits(t, cands)), range(len(cands))))[1]
        seen[id(cands[best])] = (cands[best], score(cands[best]))
    return min(seen.values(), key=lambda p: p[1])[0]


def forest_guess(t, train, test, goal=0):
    """A random forest trained on `train` and its guesses for `test`."""
    model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
    model.fit(bits(t, train), [r.cells[t.y[goal].at] for r in train])
    return model.predict(bits(t, test))
