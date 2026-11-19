"""step14, flow (players): judge scouting reports as they arrive one at a time.

    PYTHONPATH=src python3 -m domains.players.step14

The csv has no seasons, so "arrival order" is random and the drift is simulated: after a change
point every new player's penalties rating is made 3%, 8% or 15% worse, as if the intake had
weakened. The machinery is SS-W step4's.
"""
import sys

from domains.players.step11 import DATA, SKIP
from flow import report
from table import Table


def main(path=DATA, seed=1, runs=8):
    report(Table.read(path, SKIP), seed, runs, "players")


if __name__ == "__main__":
    main(*sys.argv[1:2])
