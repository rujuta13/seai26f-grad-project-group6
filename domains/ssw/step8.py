"""step8, fix: the fewest flag flips that turn a slow build into a fast one.

    PYTHONPATH=src python3 -m domains.ssw.step8

A build counts as slow if it is in the slowest 25% on speed, fast if in the fastest 25%.
Plans are made on slow builds the guesser never trained on.
"""
import random
import statistics
import sys

from blame import blame, split
from fix import index, plan
from guess import fit
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, trials=300):
    rng, t = random.Random(seed), Table.read(path)
    rows = rng.sample(t.rows, len(t.rows))
    train, test = rows[:int(.8 * len(rows))], rows[int(.8 * len(rows)):]
    tree, where, speed = fit(t, train, rng), index(t), t.y[0].at
    order = sorted(float(r.cells[speed]) for r in t.rows)
    fast, slow = order[len(order) // 4], order[3 * len(order) // 4]
    found = blame(t, *split(t, train, t.y[0]), rng)
    slow_ones = [r for r in test if float(r.cells[speed]) >= slow]
    for ask, frozen in ((15, ()), (16, ("inline",))):
        made, flips, worked = 0, [], 0
        for r in rng.sample(slow_ones, trials):
            got = plan(t, tree, where, r, found, fast, frozen)
            if got:
                made += 1
                flips.append(len(got[0]))
                worked += float(got[1].cells[speed]) <= fast
        what = "free to flip anything" if not frozen else "never touching " + ", ".join(frozen)
        print(f"ask {ask}: {what}, {trials} slow builds")
        print(f"  plan found for {made / trials:.0%}; median {statistics.median(flips)} flips (most {max(flips)}); "
              f"{worked / max(made, 1):.0%} of plans really reach the fast group")
    r = slow_ones[0]
    for frozen in ((), ("inline",)):
        got = plan(t, tree, where, r, found, fast, frozen)
        print(f"  e.g. a slow build ({float(r.cells[speed]):.4g}): " + (f"flip {', '.join(got[0])} -> {float(got[1].cells[speed]):.4g}" if got else "no plan"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
