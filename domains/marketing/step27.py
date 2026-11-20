"""step27, choose (marketing): find the best customers while test-mailing very few.

    PYTHONPATH=src python3 -m domains.marketing.step27

Same machinery as SS-W step7 and players step17. A "test mailing" is one look at a customer's goals.
"""
import random
import sys

from choose import compare, only, pick
from domains.marketing.step21 import DATA, SKIP
from table import Table


def main(path=DATA, seed=1, budget=50, repeats=40):
    t = Table.read(path, SKIP)
    names = [c.name for c in t.y]
    store, web = names.index("NUMSTOREPURCHASES+"), names.index("NUMWEBPURCHASES+")
    print(f"ask 13: best customer after {budget} test mailings, {repeats} repeats (rank 1% = better than 99% of {len(t.rows)})")
    compare(t, budget, repeats, "customers")
    b = pick(t, random.Random(seed), budget)
    print("  one run's pick: " + ", ".join(f"{c.name} {b.cells[c.at]:.0f}" for c in t.y[:4]) + f", response {b.cells[t.y[6].at]:.0f}")
    print("\nask 14: does the best choice change if store purchases matter most?")
    for name, i in (("store purchases", store), ("web purchases", web)):
        score = only(t, i)
        print(f" aiming at {name}:")
        compare(t, budget, repeats, "customers", score=score)
        b = pick(t, random.Random(seed), budget, score)
        print(f"  one run's pick: store {b.cells[t.y[store].at]:.0f}, web {b.cells[t.y[web].at]:.0f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
