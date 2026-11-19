"""step2, guess: predict a flag set's speed and CPU before benchmarking it, and say how sure.

    PYTHONPATH=src python3 -m domains.ssw.step2
"""
import sys

from guess import report
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, stop=32):
    report(Table.read(path), seed, "builds",
           lambda t, r: "on: " + (", ".join(c.name for c in t.x if r.cells[c.at] == "1") or "-"), stop)


if __name__ == "__main__":
    main(*sys.argv[1:2])
