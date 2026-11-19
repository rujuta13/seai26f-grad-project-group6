"""step11, remember (players): the types of player in the pool, and the strong penalty-takers.

    PYTHONPATH=src python3 -m domains.players.step11

`num` is a row counter and `Rank` a duplicate-laden ID, so neither describes a player. This copy
of the file has also lost position, height, weight, foot, league and nation (all zero), so
"types of player" has to come from the ratings alone.
"""
import sys

from remember import report, spread, traits
from table import Table

DATA = "data/all_players.csv"
SKIP = ("num", "Rank")


def main(path=DATA, seed=1):
    t = Table.read(path, SKIP)
    spec, at = spread(t), {c.name: c.at for c in t.cols}
    report(t, seed, lambda t, g: traits(t, g, spec),
           lambda t, r: f"{[r.cells[c.at] for c in t.y]}  OVR {r.cells[at['OVR']]:.0f}, age {r.cells[at['Age']]:.0f}",
           "players", "ratings")


if __name__ == "__main__":
    main(*sys.argv[1:2])
