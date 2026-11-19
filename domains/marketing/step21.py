"""step21, remember (marketing): the kinds of shopper we have, and the best customers.

    PYTHONPATH=src python3 -m domains.marketing.step21

The three derived totals (spend, regular-product spend, campaigns accepted) just add up other
columns, so keeping them would count the same customers' behaviour twice in every distance. The
two Z_ columns never vary and the loader drops them by itself.
"""
import sys

from remember import report, spread, traits
from table import Table

DATA = "data/Marketing_Analytics.csv"
SKIP = ("MNTTOTAL", "MNTREGULARPRODS", "ACCEPTEDCMPOVERALL")


def main(path=DATA, seed=1):
    t = Table.read(path, SKIP)
    spec, at = spread(t), {c.name: c.at for c in t.cols}
    report(t, seed, lambda t, g: traits(t, g, spec),
           lambda t, r: f"{[round(r.cells[c.at]) for c in t.y]}  income {r.cells[at['INCOME']]:.0f}, age {r.cells[at['AGE']]:.0f}",
           "customers", "columns")


if __name__ == "__main__":
    main(*sys.argv[1:2])
