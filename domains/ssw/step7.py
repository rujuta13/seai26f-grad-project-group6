"""step7, choose: find a very good build while benchmarking very few.

    PYTHONPATH=src python3 -m domains.ssw.step7
"""
import random
import sys

from choose import compare, only, pick
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, budget=30, repeats=40):
    t = Table.read(path)
    flags = lambda r: " ".join(c.name for c in t.x if r.cells[c.at] == "1") or "(none)"
    print(f"ask 13: best build after benchmarking {budget} of {len(t.rows)} ({repeats} repeats; "
          f"rank 1% = better than 99% of all builds)")
    compare(t, budget, repeats, "builds")
    print("\nask 14: does the best choice change with the goal?")
    for name, score in (("speed and CPU", None), ("speed only", only(t, 0)), ("CPU only", only(t, 1))):
        b = pick(t, random.Random(seed), budget, score)
        print(f"  {name:14} speed {b.cells[t.y[0].at]:.4g}, CPU {b.cells[t.y[1].at]:.4g}; on: {flags(b)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
