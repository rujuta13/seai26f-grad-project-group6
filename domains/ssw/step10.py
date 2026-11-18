"""step10, race: our tuner and guesser against off-the-shelf rivals.

    PYTHONPATH=src python3 -m domains.ssw.step10

Rivals: random search, and a random-forest tuner and guesser from scikit-learn. Every contender
gets the same budget. `same` (step 3) says whether a difference between two is real.
"""
import bisect
import random
import statistics
import sys

from choose import pick
from guess import fit, guess
from race import forest_guess, forest_pick
from stats import same
from table import Table

DATA = "data/SS-W.csv"


def verdict(a, b, names):
    if same(sorted(a), sorted(b)):
        return f"{names[0]} vs {names[1]}: no real difference"
    return f"{names[0]} vs {names[1]}: real, {names[0] if statistics.median(a) < statistics.median(b) else names[1]} is better"


def main(path=DATA, seed=1, budget=30, repeats=40, ntrain=500):
    t = Table.read(path)
    ranked = sorted(map(t.disty, t.rows))
    pct = lambda r: 100 * bisect.bisect_left(ranked, t.disty(r)) / len(ranked)
    tuners = {"ours": lambda rng: pick(t, rng, budget),
              "forest": lambda rng: forest_pick(t, rng, budget),
              "random": lambda rng: min(rng.sample(t.rows, budget), key=t.disty)}
    got = {k: [pct(f(random.Random(s))) for s in range(repeats)] for k, f in tuners.items()}
    print(f"ask 19: best build after {budget} benchmarks, {repeats} repeats (rank %: lower is better)")
    for k, v in got.items():
        print(f"  {k:7} median rank {statistics.median(v):.2f}%, worst {max(v):.1f}%")
    print("  " + verdict(got["ours"], got["random"], ("ours", "random")))
    print("  " + verdict(got["ours"], got["forest"], ("ours", "forest")))
    reps, errs = 10, {"ours": [], "forest": [], "mean": []}
    speed = t.y[0].at
    for s in range(reps):
        rng = random.Random(s)
        rows = rng.sample(t.rows, ntrain + 1500)
        train, test = rows[:ntrain], rows[ntrain:]
        truth = [r.cells[speed] for r in test]
        mean = statistics.fmean(r.cells[speed] for r in train)
        tree = fit(t, train, rng)
        for k, pred in (("ours", [guess(t, tree, r)[0][0] for r in test]), ("forest", forest_guess(t, train, test)),
                        ("mean", [mean] * len(test))):
            errs[k].append(statistics.fmean(abs(p - y) / y for p, y in zip(pred, truth)) * 100)
    print(f"\nask 20: predict build speed after seeing {ntrain} builds, mean error on 1500 unseen ({reps} repeats)")
    for k, v in errs.items():
        print(f"  {k:7} {statistics.median(v):.1f}% error")
    print("  " + verdict(errs["ours"], errs["forest"], ("ours", "forest")))
    print("  " + verdict(errs["ours"], errs["mean"], ("ours", "always-the-mean")))


if __name__ == "__main__":
    main(*sys.argv[1:2])
