"""step19, spot (players): doubtful ratings and unusual players.

    PYTHONPATH=src python3 -m domains.players.step19

The csv has no data-entry mistakes we know of, so for ask 17 we make some: a penalties rating
typed with its digits swapped (71 as 17), left as 0, or typed as 99.
"""
import random
import sys

from domains.players.step11 import DATA, SKIP
from remember import spread, traits
from spot import catch, changed, cutoff, nearest, odd
from table import Table


def swapped(t, row):
    v = int(row.cells[t.y[0].at])
    new = int(str(v)[::-1])
    return changed(t, row, float(new)) if v >= 10 and new != v else None


def main(path=DATA, seed=1, n=300):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    print(f"ask 17: flag penalties ratings that look like typing mistakes ({n} players each)")
    catch(t, seed, n, [("digits swapped", lambda r: swapped(t, r)),
                       ("typed as 0", lambda r: changed(t, r, 0.0) if r.cells[t.y[0].at] else None),
                       ("typed as 99", lambda r: changed(t, r, 99.0) if r.cells[t.y[0].at] < 90 else None)])
    seen, spec = rng.sample(t.rows, 500), spread(t)
    cut, cands = cutoff(t, seen), rng.sample(t.rows, 300)
    far = sorted(cands, key=lambda r: -nearest(t, seen, r))
    print(f"\nask 18: players further from every one of 500 known players than they are from each other "
          f"({len([r for r in cands if odd(t, seen, r, cut)]) / len(cands):.0%} of 300 new players)")
    for r in far[:3]:
        print("  " + traits(t, [r], spec, k=4))


if __name__ == "__main__":
    main(*sys.argv[1:2])
