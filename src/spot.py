"""Spot: results that look like glitches, and rows unlike anything seen before."""
from guess import guess


def glitch(t, tree, row, k=3):
    """Names of the goals whose measured value lies far outside what rows like this one gave:
    more than `k` interquartile ranges beyond the quartiles of the row's leaf."""
    out = []
    for c, (_, q1, q3) in zip(t.y, guess(t, tree, row, 0.25, 0.75)):
        v, span = row.cells[c.at], q3 - q1
        if v != "?" and (v < q1 - k * span or v > q3 + k * span):
            out.append(c.name)
    return out


def nearest(t, seen, row, skip=None):
    return min(t.distx(row, s) for s in seen if s is not skip)


def cutoff(t, seen, q=0.95):
    """How far a seen row usually is from its nearest other seen row: the `q` quantile."""
    d = sorted(nearest(t, seen, s, s) for s in seen)
    return d[int(q * (len(d) - 1))]


def odd(t, seen, row, cut):
    """True if `row` is farther from everything seen than seen rows usually are from each other."""
    return nearest(t, seen, row) > cut
