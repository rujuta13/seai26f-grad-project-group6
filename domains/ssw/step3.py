"""step3, certify: is a gap real, or run-to-run noise?

    PYTHONPATH=src python3 -m domains.ssw.step3

Both comparisons use 100 random builds per side, the number a team might benchmark: `same`
is built for modest samples, and on thousands of rows it calls every tiny gap different.
"""
import random
import sys

from certify import gap, say
from cluster import families
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, n=100):
    rng, t = random.Random(seed), Table.read(path)
    print(f"ask 5: is each flag's effect real? (on vs off, {n} builds each; lower is better)")
    print(f"  {'flag':26} " + " ".join(f"{c.name:34}" for c in t.y))
    for c in t.x:
        on, off = ([r for r in t.rows if r.cells[c.at] == v] for v in "10")
        print(f"  {c.name:26} " + " ".join(f"{say(g, 'on'):34}" for g in gap(t, rng.sample(on, n), rng.sample(off, n))))
    a, b = sorted(families(t, random.Random(seed)), key=lambda g: sum(map(t.disty, g)) / len(g))[:2]
    flags = lambda g: sum(sum(r.cells[c.at] == "1" for c in t.x) for r in g) / len(g)
    print(f"\nask 6: our two best flag-set families ({len(a)} and {len(b)} builds), {n} builds each")
    gaps = gap(t, rng.sample(a, n), rng.sample(b, n))
    for c, g in zip(t.y, gaps):
        print(f"  {c.name:14} {say(g, 'first')}")
    if all(g[2] for g in gaps):
        simple = "first" if flags(a) <= flags(b) else "second"
        print(f"  no real difference: take the simpler one ({simple}, {min(flags(a), flags(b)):.1f} flags on "
              f"vs {max(flags(a), flags(b)):.1f})")
    else:
        print("  they differ: the first family is the better one" if gaps[0][0] < gaps[0][1] else "  they differ")


if __name__ == "__main__":
    main(*sys.argv[1:2])
