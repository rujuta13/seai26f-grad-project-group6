"""step16, justify (players): reasons a manager, or a video analyst, can check.

    PYTHONPATH=src python3 -m domains.players.step16

The reasons are the columns step 15's `blame` found really separate the best players from the
worst (by both goals together), and where this player sits against the pool average.
"""
import bisect
import random
import statistics
import sys

from blame import blame
from domains.players.step11 import DATA, SKIP
from domains.players.step13 import predicted
from guess import fit
from justify import one_line
from table import Table


def main(path=DATA, seed=1):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    ranked = sorted(t.rows, key=t.disty)
    k = len(ranked) // 10
    found = blame(t, ranked[-k:], ranked[:k], rng)
    median = statistics.median(map(t.disty, t.rows))
    rows = rng.sample(t.rows, len(t.rows))
    train, test = rows[:len(rows) * 4 // 5], rows[len(rows) * 4 // 5:]
    tree = fit(t, train, rng)
    pick = min(test, key=lambda r: predicted(t, tree, r))
    pct = lambda r, c: bisect.bisect_left(sorted(x.cells[c.at] for x in t.rows), r.cells[c.at]) / len(t.rows)
    print("ask 11: why sign the player our guesser likes best of the unseen ones")
    print("  his goals: " + ", ".join(f"{c.name} {pick.cells[c.at]:.0f} (better than {pct(pick, c):.0%} of players)" for c in t.y))
    print("  " + one_line(t, pick, found, median, k=5))
    print("\nask 12: four players rated low, with reasons to check on video")
    for r in [r for r in rng.sample(t.rows, 40) if t.disty(r) > median][:4]:
        print("  " + one_line(t, r, found, median))


if __name__ == "__main__":
    main(*sys.argv[1:2])
