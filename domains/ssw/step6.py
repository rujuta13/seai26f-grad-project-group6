"""step6, justify: reasons a skeptic can check.

    PYTHONPATH=src python3 -m domains.ssw.step6
"""
import random
import statistics
import sys

from blame import blame, split
from certify import gap
from justify import one_line, recommend
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, n=100):
    rng, t = random.Random(seed), Table.read(path)
    print("ask 11: the recommended flag set, one checked reason per flag")
    rec = recommend(t, rng, n)
    for name, setting, why in rec:
        print(f"  {name:28} {setting:3}  {why}")
    chosen = [r for r in t.rows if all(r.cells[c.at] == ("1" if s == "on" else "0")
                                       for c, (_, s, why) in zip(t.x, rec) if "no real" not in why)]
    print(f"\n  check: {len(chosen)} builds follow every flag that matters; against {n} random builds:")
    for goal, (a, b, noise) in zip(t.y, gap(t, rng.sample(chosen, min(n, len(chosen))), rng.sample(t.rows, n))):
        print(f"    {goal.name:14} median {a:.4g} vs {b:.4g} ({'noise' if noise else 'real'})")
    print("\nask 12: one sentence per build")
    found = blame(t, *split(t, t.rows, t.y[0]), rng)
    median = statistics.median(map(t.disty, t.rows))
    for r in rng.sample(t.rows, 4):
        print("  " + one_line(t, r, found, median))


if __name__ == "__main__":
    main(*sys.argv[1:2])
