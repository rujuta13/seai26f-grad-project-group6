"""step29, spot (marketing): doubtful customer records and customers unlike anyone else.

    PYTHONPATH=src python3 -m domains.marketing.step29

The file is already clean (ages 24-80, no wild incomes), so for ask 17 we make the mistakes: an age typed
as 0 or with an extra digit, an income with an extra zero, a store-purchases count typed as 99.
Inputs are checked against fences from clean rows (`fences`); purchases by `glitch`, as in players step19.
"""
import random
import sys

from domains.marketing.step21 import DATA, SKIP
from remember import spread, traits
from spot import catch, changed, cutoff, fences, nearest, odd, outside
from table import Row, Table


def typo(t, row, name, make):
    """The row with input column `name` replaced by make(old value), or None if nothing changes."""
    c = next(c for c in t.cols if c.name == name)
    new = make(row.cells[c.at])
    if new == row.cells[c.at]:
        return None
    cells = list(row.cells)
    cells[c.at] = new
    return Row(cells, row.x, row.y)


def main(path=DATA, seed=1, n=300):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    store = [c.name for c in t.y].index("NUMSTOREPURCHASES+")
    print(f"ask 17: flag customer records that look wrong ({n} customers each)")
    rows = rng.sample(t.rows, len(t.rows))
    fence, test = fences(t, rows[:len(rows) * 4 // 5], ("AGE", "INCOME"), 1.5), rows[len(rows) * 4 // 5:][:n]
    print(f"  clean records flagged on age or income: {sum(bool(outside(t, r, fence)) for r in test) / n:.1%}")
    for label, name, make in (("age typed with an extra digit", "AGE", lambda v: v * 10),
                              ("age typed as 0", "AGE", lambda v: 0.0),
                              ("income with an extra zero", "INCOME", lambda v: v * 10)):
        bad = [m for m in (typo(t, r, name, make) for r in test) if m]
        print(f"  {label}: flagged {sum(bool(outside(t, m, fence)) for m in bad) / len(bad):.0%} of {len(bad)}")
    print(" store purchases:")
    catch(t, seed, n, [("typed as 99", lambda r: changed99(t, r, store))], floor=2)
    seen, spec = rng.sample(t.rows, 500), spread(t)
    cut, cands = cutoff(t, seen), rng.sample(t.rows, 300)
    far = sorted(cands, key=lambda r: -nearest(t, seen, r))
    print(f"\nask 18: customers further from every one of 500 known customers than they are from each other "
          f"({len([r for r in cands if odd(t, seen, r, cut)]) / len(cands):.0%} of 300 new customers)")
    for r in far[:3]:
        print("  " + traits(t, [r], spec, k=4))
    print("  (unusual is not the same as fraud: the file has no fraud label to check against)")


def changed99(t, row, goal):
    return changed(t, row, 99.0, goal) if row.cells[t.y[goal].at] < 90 else None


if __name__ == "__main__":
    main(*sys.argv[1:2])
