"""step4, flow: judge benchmark results as they arrive one at a time.

    PYTHONPATH=src python3 -m domains.ssw.step4

The csv has no dates, so "nightly" is a random order, and the drift is simulated: after a
change point every build is made to look `slowdown` slower, as if the machine had degraded.
"""
import random
import statistics
import sys

from flow import Keeper, Watch
from table import Row, Table

DATA = "data/SS-W.csv"


def slower(t, row, k):
    cells = list(row.cells)
    cells[t.y[0].at] *= 1 + k
    return Row(cells, row.x, tuple(c.norm(cells[c.at]) for c in t.y))


def keeping(t, rng, share=0.05):
    keeper = Keeper(t, share)
    cut = sorted(map(t.disty, t.rows))[int(share * len(t.rows))]
    kept = [(i, t.disty(r) <= cut) for i, r in enumerate(rng.sample(t.rows, len(t.rows))) if keeper.add(r)]
    for label, cap in (("first 500 arrivals", 500), ("all arrivals", len(t.rows))):
        hits = [good for i, good in kept if i < cap]
        print(f"  {label:18} kept {len(hits):5d}, {sum(hits) / len(hits):.0%} truly in the best {share:.0%}")
    print(f"  caught {sum(g for _, g in kept) / int(share * len(t.rows)):.0%} of all the best builds")


def watching(t, slowdown, seed, change=1500, total=3000, n=100):
    rng = random.Random(seed)
    stream = rng.sample(t.rows, n * 5 + total)
    watch, early, late = Watch(t, rng.sample(stream[:n * 5], n), n), 0, None
    for i, r in enumerate(stream[n * 5:]):
        if watch.add(slower(t, r, slowdown) if i >= change else r):
            if i < change:
                early += 1
            elif late is None:
                late = i - change
    return early, late


def main(path=DATA, seed=1, runs=8):
    rng, t = random.Random(seed), Table.read(path)
    print("ask 7: keep a build only if it is in the best 5% seen so far")
    keeping(t, rng)
    print(f"\nask 8: warn when results stop looking like the reference month ({runs} simulated nights each)")
    print(f"  {'slowdown':>8} {'false alarms per 1500 builds':>30} {'caught':>7} {'median delay':>13}")
    for k in (0, 0.03, 0.08, 0.15):
        res = [watching(t, k, s) for s in range(runs)]
        lags = [late for _, late in res if late is not None]
        print(f"  {k:8.0%} {sum(e for e, _ in res) / runs:30.1f} {len(lags):>5}/{runs} "
              f"{(str(round(statistics.median(lags))) + ' builds') if lags else '-':>13}")
    print("  (at 0% nothing changes, so any 'caught' is a false alarm)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
