"""step1, remember: the families LLVM flag sets fall into, and the unusually fast, light ones.

    PYTHONPATH=src python3 -m domains.ssw.step1
"""
import sys

from remember import report
from table import Table

DATA = "data/SS-W.csv"


def on(t, rows, share=0.8):
    """Flags switched on in at least `share` of these rows."""
    return [c.name for c in t.x if sum(r.cells[c.at] == "1" for r in rows) >= share * len(rows)]


def main(path=DATA, seed=1):
    report(Table.read(path), seed,
           lambda t, g: "always on: " + (", ".join(on(t, g)) or "-"),
           lambda t, r: f"{[r.cells[c.at] for c in t.y]}  on: {', '.join(c.name for c in t.x if r.cells[c.at] == '1')}",
           "builds", "flags")


if __name__ == "__main__":
    main(*sys.argv[1:2])
