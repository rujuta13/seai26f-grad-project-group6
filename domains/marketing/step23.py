"""step23, certify (marketing): is a gap between groups of customers real, or luck?

    PYTHONPATH=src python3 -m domains.marketing.step23

Same machinery as SS-W step3 and players step13: 100 customers from each side, `same` on every goal.
Ask 5 can only compare customers who answered the last campaign with those who did not. The
file has no before-and-after, so a real gap there is an association, not proof the campaign caused it.
"""
import random
import sys

from certify import gap, say
from domains.marketing.step21 import DATA, SKIP
from table import Table


def versus(t, rng, a, b, who, n=100):
    for c, g in zip(t.y, gap(t, rng.sample(a, n), rng.sample(b, n))):
        print(f"  {c.name:22} {say(g, who)}")


def main(path=DATA, seed=1):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    at = {c.name: c.at for c in t.cols}
    split = lambda col, test: ([r for r in t.rows if test(r.cells[at[col]])], [r for r in t.rows if not test(r.cells[at[col]])])
    yes, no = split("RESPONSE+", lambda v: v == 1)
    print(f"ask 5: customers who answered the last campaign ({len(yes)}) vs those who did not ({len(no)}), 100 each")
    versus(t, rng, yes, no, "responders")
    teens, none = split("TEENHOME", lambda v: v >= 1)
    print(f"\nask 6: homes with teenagers ({len(teens)}) vs without ({len(none)}), 100 each")
    versus(t, rng, teens, none, "teen homes")


if __name__ == "__main__":
    main(*sys.argv[1:2])
