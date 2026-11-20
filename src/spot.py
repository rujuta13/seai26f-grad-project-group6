"""Spot: results that look like glitches, and rows unlike anything seen before."""
import random

from guess import fit, guess
from table import Row


def glitch(t, tree, row, k=3, floor=0):
    """Names of the goals whose measured value lies far outside what rows like this one gave:
    more than `k` interquartile ranges beyond the quartiles of the row's leaf (a range narrower than `floor`
    counts as `floor`, for counts where most leaves agree exactly)."""
    out = []
    for c, (_, q1, q3) in zip(t.y, guess(t, tree, row, 0.25, 0.75)):
        v, span = row.cells[c.at], max(q3 - q1, floor)
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


def changed(t, row, v, goal=0):
    """The row with one goal's measured value replaced by `v`."""
    cells = list(row.cells)
    cells[t.y[goal].at] = v
    return Row(cells, row.x, row.y)


def catch(t, seed, n, mistakes, floor=0):
    """Fit on 80% of the rows, then flag glitches among `n` unseen ones: clean, and again after each
    (label, make) mistake. `make(row)` returns the damaged row, or None if it would not change."""
    rng = random.Random(seed)
    rows = rng.sample(t.rows, len(t.rows))
    tree, clean = fit(t, rows[:len(rows) * 4 // 5], rng), rng.sample(rows[len(rows) * 4 // 5:], n)
    print(f"  clean results flagged: {sum(bool(glitch(t, tree, r, floor=floor)) for r in clean) / n:.1%}")
    for label, make in mistakes:
        bad = [m for m in map(make, clean) if m]
        print(f"  {label}: flagged {sum(bool(glitch(t, tree, m, floor=floor)) for m in bad) / len(bad):.0%} of {len(bad)}")


def fences(t, rows, names, k=3):
    """For each named input column, the (low, high) limits `k` interquartile ranges beyond the quartiles of `rows`."""
    out = {}
    for c in t.cols:
        if c.name in names:
            v = sorted(r.cells[c.at] for r in rows)
            q1, q3 = v[len(v) // 4], v[3 * len(v) // 4]
            out[c.name] = (q1 - k * (q3 - q1), q3 + k * (q3 - q1))
    return out


def outside(t, row, fence):
    """Names of the input columns whose value lies outside its limits."""
    at = {c.name: c.at for c in t.cols}
    return [n for n, (lo, hi) in fence.items() if not lo <= row.cells[at[n]] <= hi]
