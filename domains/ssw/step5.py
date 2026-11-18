"""step5, blame: what makes the bad builds bad?

    PYTHONPATH=src python3 -m domains.ssw.step5

A flag is blamed when it is on much more (+) or much less (-) often among the bad builds
than among the good ones, and `same` agrees the difference is real.
"""
import random
import sys

from blame import blame, split
from table import Table

DATA = "data/SS-W.csv"


def show(found, top):
    for name, d, real in [f for f in found if f[2]][:top]:
        print(f"  {name:28} {'on' if d > 0 else 'off'} {abs(d) * 100:.0f} points more often in the bad builds")


def main(path=DATA, seed=1, top=5):
    rng, t = random.Random(seed), Table.read(path)
    speed, cpu = t.y
    bad, good = split(t, t.rows, speed)
    print(f"ask 9: slowest 10% ({len(bad)} builds) vs fastest 10%, flags to blame")
    show(blame(t, bad, good, rng), top)
    fast = split(t, t.rows, speed, 0.25)[1]
    bad, good = split(t, fast, cpu, 0.2)
    print(f"\nask 10: among the fastest 25% ({len(fast)} builds), heaviest-CPU 20% vs lightest 20%")
    show(blame(t, bad, good, rng), top)


if __name__ == "__main__":
    main(*sys.argv[1:2])
