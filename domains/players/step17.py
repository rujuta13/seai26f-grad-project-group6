"""step17, choose (players): find a very good player while scouting very few.

    PYTHONPATH=src python3 -m domains.players.step17

Same machinery as SS-W step7. A "scouting trip" is one look at a player's goals.
"""
import random
import sys

from choose import compare, pick
from domains.players.step11 import DATA, SKIP
from table import Table


def main(path=DATA, seed=1, budget=20, repeats=40):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    age = next(c.at for c in t.cols if c.name == "Age")
    young = [r for r in t.rows if r.cells[age] < 24]
    for ask, what, rows in ((13, "anyone", t.rows), (14, "under 24", young)):
        print(f"ask {ask}: best player ({what}, {len(rows)} to choose from) after {budget} scouting trips, "
              f"{repeats} repeats (rank 1% = better than 99% of them)")
        compare(t, budget, repeats, "players", rows)
        b = pick(t, random.Random(seed), budget, rows=rows)
        print(f"  one run's pick: " + ", ".join(f"{c.name} {b.cells[c.at]:.0f}" for c in t.y) + f", age {b.cells[age]:.0f}\n")


if __name__ == "__main__":
    main(*sys.argv[1:2])
