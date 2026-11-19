"""step12, guess (players): guess a player's penalties and strength from his other ratings.

    PYTHONPATH=src python3 -m domains.players.step12

Same machinery as SS-W step2: this file only says which table and how to show a player.
"""
import sys

from domains.players.step11 import DATA, SKIP
from guess import report
from table import Table


def who(t, r):
    at = {c.name: c.at for c in t.cols}
    return f"player with OVR {r.cells[at['OVR']]:.0f}, age {r.cells[at['Age']]:.0f}"


def main(path=DATA, seed=1):
    report(Table.read(path, SKIP), seed, "players", who)


if __name__ == "__main__":
    main(*sys.argv[1:2])
