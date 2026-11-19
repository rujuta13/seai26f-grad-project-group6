"""Rows from MOOT-style csv files, using the header rules ezr uses.

An uppercase first letter makes a column numeric, anything else symbolic. A trailing "+" or
"-" marks a goal to maximise or minimise, a trailing "X" means skip the column, and "?" is a
missing value.
"""
import csv


class Col:
    def __init__(self, at, name):
        self.at, self.name = at, name
        self.goal = name[-1] in "+-"
        self.num = name[0].isupper() or self.goal
        self.skip = name[-1] == "X"
        self.heaven = 1 if name[-1] == "+" else 0
        self.lo, self.hi = 1e32, -1e32

    def read(self, s):
        return float(s) if self.num and s != "?" else s

    def add(self, v):
        if self.num and v != "?":
            self.lo, self.hi = min(self.lo, v), max(self.hi, v)

    def norm(self, v):
        if v == "?" or not self.num:
            return v
        return (v - self.lo) / (self.hi - self.lo + 1e-32)


class Row:
    def __init__(self, cells, x, y):
        self.cells, self.x, self.y = cells, x, y  # raw cells; normalised x and goal values


class Table:
    def __init__(self, names, skip=()):
        self.cols = [Col(i, s) for i, s in enumerate(names)]
        for c in self.cols:
            c.skip = c.skip or c.name in skip
        self.x = [c for c in self.cols if not c.goal and not c.skip]
        self.y = [c for c in self.cols if c.goal]
        self.rows = []

    @classmethod
    def read(cls, path, skip=()):
        """Read a csv. Columns named in `skip`, and columns holding one value throughout (they tell
        us nothing), are left out of the inputs."""
        with open(path, newline="") as f:
            lines = csv.reader(f)
            t = cls(next(lines), skip)
            raw = [[c.read(s.strip()) for c, s in zip(t.cols, r)] for r in lines if r]
        for r in raw:
            for c in t.cols:
                c.add(r[c.at])
        t.x = [c for c in t.x if len({r[c.at] for r in raw}) > 1]
        t.rows = [Row(r, tuple(c.norm(r[c.at]) for c in t.x), tuple(c.norm(r[c.at]) for c in t.y))
                  for r in raw]
        return t

    def distx(self, a, b):
        """Distance between two rows, looking only at the independent columns (0 to 1)."""
        t = 0
        for u, v in zip(a.x, b.x):
            if "?" in (u, v):
                d = 1
            elif isinstance(u, str):
                d = u != v
            else:
                d = abs(u - v)
            t += d * d
        return (t / len(a.x)) ** 0.5

    def disty(self, row):
        """Distance from a row's goals to heaven, the best value of every goal (0 is best)."""
        t = sum(1 if v == "?" else (c.heaven - v) ** 2 for c, v in zip(self.y, row.y))
        return (t / len(self.y)) ** 0.5
