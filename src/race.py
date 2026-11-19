"""Rivals built from scikit-learn, to race our own tuner and guesser against."""
import bisect
import random
import statistics

from sklearn.ensemble import RandomForestRegressor

from choose import pick
from guess import fit, guess
from stats import same


def bits(t, rows):
    """The input columns as numbers for scikit-learn: ratings as they are, symbolic flags as 0 or 1."""
    return [[(0.0 if r.cells[c.at] == "?" else r.cells[c.at]) if c.num else float(r.cells[c.at] == "1")
             for c in t.x] for r in rows]


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


def verdict(a, b, names):
    """`same` (step 3) on two lists of scores, in words."""
    if same(sorted(a), sorted(b)):
        return f"{names[0]} vs {names[1]}: no real difference"
    return f"{names[0]} vs {names[1]}: real, {names[0] if statistics.median(a) < statistics.median(b) else names[1]} is better"


def tuner_race(t, budget, repeats, things):
    """Our tuner, the forest tuner and random picking, each given `budget` looks, `repeats` times."""
    ranked = sorted(map(t.disty, t.rows))
    pct = lambda r: 100 * bisect.bisect_left(ranked, t.disty(r)) / len(ranked)
    tuners = {"ours": lambda rng: pick(t, rng, budget),
              "forest": lambda rng: forest_pick(t, rng, budget),
              "random": lambda rng: min(rng.sample(t.rows, budget), key=t.disty)}
    got = {k: [pct(f(random.Random(s))) for s in range(repeats)] for k, f in tuners.items()}
    print(f"best {things[:-1]} after {budget} looks, {repeats} repeats (rank %: lower is better)")
    for k, v in got.items():
        print(f"  {k:7} median rank {statistics.median(v):.2f}%, worst {max(v):.1f}%")
    print("  " + verdict(got["ours"], got["random"], ("ours", "random")))
    print("  " + verdict(got["ours"], got["forest"], ("ours", "forest")))


def guess_race(t, ntrain, reps, things, goal=0, ntest=1500):
    """Our guesser, the forest and always-the-mean, each trained on `ntrain` rows and scored on `ntest` unseen."""
    errs, at = {"ours": [], "forest": [], "mean": []}, t.y[goal].at
    for s in range(reps):
        rng = random.Random(s)
        rows = rng.sample(t.rows, ntrain + ntest)
        train, test = rows[:ntrain], rows[ntrain:]
        truth, mean = [r.cells[at] for r in test], statistics.fmean(r.cells[at] for r in train)
        tree = fit(t, train, rng)
        for k, pred in (("ours", [guess(t, tree, r)[goal][0] for r in test]), ("forest", forest_guess(t, train, test, goal)),
                        ("mean", [mean] * len(test))):
            errs[k].append(statistics.fmean(abs(p - y) / y for p, y in zip(pred, truth)) * 100)
    print(f"predict {t.y[goal].name} after seeing {ntrain} {things}, mean error on {ntest} unseen ({reps} repeats)")
    for k, v in errs.items():
        print(f"  {k:7} {statistics.median(v):.1f}% error")
    print("  " + verdict(errs["ours"], errs["forest"], ("ours", "forest")))
    print("  " + verdict(errs["ours"], errs["mean"], ("ours", "always-the-mean")))
