"""step7, choose: find a very good build while benchmarking very few.

    PYTHONPATH=src python3 -m domains.ssw.step7
"""
import bisect
import random
import statistics
import sys

from choose import only, pick
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, budget=30, repeats=40):
    t = Table.read(path)
    flags = lambda r: " ".join(c.name for c in t.x if r.cells[c.at] == "1") or "(none)"
    print(f"ask 13: best build after benchmarking {budget} of {len(t.rows)} ({repeats} repeats; "
          f"rank 1% = better than 99% of all builds)")
    ranked = sorted(map(t.disty, t.rows))
    pct = lambda r: bisect.bisect_left(ranked, t.disty(r)) / len(ranked)
    for name, find in (("tuner ", lambda rng: pick(t, rng, budget)),
                       ("random", lambda rng: min(random.Random(rng.random()).sample(t.rows, budget), key=t.disty))):
        got = [pct(find(random.Random(s))) for s in range(repeats)]
        print(f"  {name} median rank {statistics.median(got):.1%}, worst of {repeats} runs {max(got):.1%}")
    print("\nask 14: does the best choice change with the goal?")
    for name, score in (("speed and CPU", None), ("speed only", only(t, 0)), ("CPU only", only(t, 1))):
        b = pick(t, random.Random(seed), budget, score)
        print(f"  {name:14} speed {b.cells[t.y[0].at]:.4g}, CPU {b.cells[t.y[1].at]:.4g}; on: {flags(b)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
