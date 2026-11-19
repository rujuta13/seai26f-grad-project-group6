"""step9, spot: glitches and oddballs.

    PYTHONPATH=src python3 -m domains.ssw.step9

Ask 17 needs glitches to test on, and the csv has none, so we make them: some clean results are
rewritten as if the benchmark had run 0.5x, 2x or 4x too slow (a stuck run, a noisy neighbour).
"""
import random
import sys

from cluster import families
from spot import catch, changed, cutoff, odd
from table import Table

DATA = "data/SS-W.csv"


def scaled(t, row, k):
    return changed(t, row, row.cells[t.y[0].at] * k)


def main(path=DATA, seed=1, n=300):
    rng, t = random.Random(seed), Table.read(path)
    print(f"ask 17: flag glitches ({n} results each; clean ones should not be flagged)")
    catch(t, seed, n, [(f"speed measured {k}x", lambda r, k=k: scaled(t, r, k)) for k in (0.5, 2, 4)])
    gs = sorted(families(t, random.Random(seed)), key=lambda g: sum(map(t.disty, g)) / len(g))
    seen = rng.sample(gs[0], 400)
    cut = cutoff(t, seen)
    print(f"\nask 18: history is 400 builds from our best family; which flag sets are unlike anything in it?")
    print(f"  same family, new builds flagged: {sum(odd(t, seen, r, cut) for r in rng.sample(gs[0], 150)) / 150:.0%}")
    for g in gs[1:4]:
        print(f"  another family ({len(g)} builds) flagged: {sum(odd(t, seen, r, cut) for r in rng.sample(g, 150)) / 150:.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
