"""Judge rows one at a time, as they arrive."""
from bisect import bisect_left, insort
from collections import deque

from certify import gap


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
