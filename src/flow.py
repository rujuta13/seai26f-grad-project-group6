"""Judge rows one at a time, as they arrive."""
import random
import statistics
from bisect import bisect_left, insort
from collections import deque

from certify import gap
from table import Row


class Keeper:
    """Keep a row if its distance to heaven is in the best `share` of the rows seen so far."""
    def __init__(self, t, share=0.05, warm=50):
        self.t, self.share, self.warm, self.seen = t, share, warm, []

    def add(self, row):
        d = self.t.disty(row)
        insort(self.seen, d)
        return len(self.seen) > self.warm and bisect_left(self.seen, d) < self.share * len(self.seen)


class Watch:
    """Warn when the latest `n` rows stop looking like a reference sample, as `same` judges it."""
    def __init__(self, t, ref, n=100, every=10, patience=5):
        self.t, self.ref, self.every, self.patience = t, ref, every, patience
        self.recent, self.seen, self.run = deque(maxlen=n), 0, 0

    def add(self, row):
        """True when `patience` checks in a row, `every` rows apart, all say things changed."""
        self.recent.append(row)
        self.seen += 1
        if len(self.recent) < self.recent.maxlen or self.seen % self.every:
            return False
        self.run = 0 if all(g[2] for g in gap(self.t, self.ref, self.recent)) else self.run + 1
        if self.run >= self.patience:
            self.run = 0
            return True
        return False


def worse(t, row, k):
    """The row with its first goal made `k` worse, as if the machine, or the market, had shifted."""
    cells, c = list(row.cells), t.y[0]
    cells[c.at] *= 1 + k if c.heaven == 0 else 1 - k
    return Row(cells, row.x, tuple(c.norm(cells[c.at]) for c in t.y))


def keeping(t, rng, share=0.05):
    """Print how well `Keeper` does on a shuffled stream of every row."""
    keeper = Keeper(t, share)
    cut = sorted(map(t.disty, t.rows))[int(share * len(t.rows))]
    kept = [(i, t.disty(r) <= cut) for i, r in enumerate(rng.sample(t.rows, len(t.rows))) if keeper.add(r)]
    for label, cap in (("first 500 arrivals", 500), ("all arrivals", len(t.rows))):
        hits = [good for i, good in kept if i < cap]
        print(f"  {label:18} kept {len(hits):5d}, {sum(hits) / len(hits):.0%} truly in the best {share:.0%}")
    print(f"  caught {sum(g for _, g in kept) / int(share * len(t.rows)):.0%} of all the best rows")


def watching(t, slowdown, seed, change=1500, total=3000, n=100):
    rng = random.Random(seed)
    stream = rng.sample(t.rows, n * 5 + total)
    watch, early, late = Watch(t, rng.sample(stream[:n * 5], n), n), 0, None
    for i, r in enumerate(stream[n * 5:]):
        if watch.add(worse(t, r, slowdown) if i >= change else r):
            if i < change:
                early += 1
            elif late is None:
                late = i - change
    return early, late


def report(t, seed, runs, things):
    rng = random.Random(seed)
    print(f"ask: keep a row only if it is in the best 5% of the {things} seen so far")
    keeping(t, rng)
    print(f"\nask: warn when results stop looking like the reference sample ({runs} simulated runs each)")
    print(f"  {'worsening':>9} {'false alarms per 1500':>22} {'caught':>7} {'median delay':>13}")
    for k in (0, 0.03, 0.08, 0.15):
        res = [watching(t, k, s) for s in range(runs)]
        lags = [late for _, late in res if late is not None]
        print(f"  {k:9.0%} {sum(e for e, _ in res) / runs:22.1f} {len(lags):>5}/{runs} "
              f"{(str(round(statistics.median(lags))) + ' rows') if lags else '-':>13}")
    print("  (at 0% nothing changes, so any 'caught' is a false alarm)")
