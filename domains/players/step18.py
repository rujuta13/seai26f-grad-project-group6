"""step18, fix (players): the smallest training focus that should lift a player's penalties.

    PYTHONPATH=src python3 -m domains.players.step18

For a player short of a target, take the three ratings (from step 15's `blame`) where he trails the
good players most, and lift each to those players' median. Then check the plan the only honest way
available: do real players with the changed ratings, found in a set of players the plan never
used, have good penalties? That is evidence of association, not a coaching promise.
"""
import random
import sys

from blame import blame, split
from domains.players.step11 import DATA, SKIP
from fix import evaluate
from table import Table

# Ratings a coach cannot train: age, and the overall and six headline ratings that are built from the others.
FROZEN = ("Age", "OVR", "PAC", "SHO", "PAS", "DRI", "DEF", "PHY")


def pen(t, r):
    return r.cells[t.y[0].at]


def main(path=DATA, seed=1, trials=60):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    rows = rng.sample(t.rows, len(t.rows))
    ref, rest = rows[:2000], rows[2000:]
    found = blame(t, *split(t, rest, t.y[0]), rng)
    v = sorted(pen(t, r) for r in rest)
    at = lambda q: v[int(q * len(v))]
    for ask, who, lo, hi, q in ((15, "close to our shortlist (70th-85th percentile)", .70, .85, .85),
                                (16, "weak (25th-40th percentile)", .25, .40, .75)):
        want, good = at(q), [r for r in rest if pen(t, r) >= at(q)]
        pool = [r for r in rest[:5000] if at(lo) <= pen(t, r) < at(hi)]
        evaluate(t, rng, pool, ref, found, good, want, FROZEN, ask, who, "players", trials=trials)


if __name__ == "__main__":
    main(*sys.argv[1:2])
