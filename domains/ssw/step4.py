"""step4, flow: judge benchmark results as they arrive one at a time.

    PYTHONPATH=src python3 -m domains.ssw.step4

The csv has no dates, so "nightly" is a random order, and the drift is simulated: after a
change point every build is made 3%, 8% or 15% slower, as if the machine had degraded.
"""
import sys

from flow import report
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, runs=8):
    report(Table.read(path), seed, runs, "builds")


if __name__ == "__main__":
    main(*sys.argv[1:2])
