"""step15, blame (players): what the weak penalty-takers lack, and what strong players have.

    PYTHONPATH=src python3 -m domains.players.step15

Same machinery as SS-W step5: contrast the two tails of a goal, column by column, and keep only
differences `same` calls real. Gaps are shown in rating points (means of the two groups).
"""
import random
import sys

from blame import blame, split
from domains.players.step11 import DATA, SKIP
from table import Table


def show(t, found, bad, good, label, top=6):
    mean = lambda g, c: sum(r.cells[c.at] for r in g) / len(g)
    at = {c.name: c for c in t.x}
    for name, d, real in [f for f in found if f[2]][:top]:
        print(f"  {name:22} {label[0]} {mean(bad, at[name]):5.1f}   {label[1]} {mean(good, at[name]):5.1f}")


def main(path=DATA, seed=1, share=0.1):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    pen, strength = t.y
    bad, good = split(t, t.rows, pen, share)
    print(f"ask 9: the weakest {share:.0%} from the spot ({len(bad)} players) vs the best {share:.0%}: what are they missing?")
    show(t, blame(t, bad, good, rng), bad, good, ("weak", "best"))
    rest, strong = split(t, t.rows, strength, share)
    print(f"\nask 10: the strongest {share:.0%} ({len(strong)} players) vs the weakest {share:.0%}: what separates them?")
    show(t, blame(t, rest, strong, rng), rest, strong, ("weak", "strong"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
