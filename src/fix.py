"""Fix: the fewest changes that turn a bad row into a good one."""
import statistics

from guess import guess, near
from justify import averages
from table import Row


def index(t):
    """Find the table row with a given setting of the input columns."""
    return {tuple(r.cells[c.at] for c in t.x): r for r in t.rows}


def plan(t, tree, where, row, found, want, frozen=(), goal=0):
    """Flip the columns `blame` found real, biggest first (never the `frozen` ones), until the
    guess for the changed row reaches `want` on `goal`. Returns (flips, changed row) or None."""
    cells, flips = [row.cells[c.at] for c in t.x], []
    pos = {c.name: i for i, c in enumerate(t.x)}
    for name, d, real in sorted(found, key=lambda f: -abs(f[1])):
        i = pos[name]
        if not real or name in frozen or (cells[i] == "1") != (d > 0):
            continue
        cells[i] = "0" if cells[i] == "1" else "1"
        flips.append(name)
        new = where.get(tuple(cells))
        if new and guess(t, tree, new)[goal][0] <= want:
            return flips, new
    return None


def improve(t, row, found, good, frozen=(), most=3, floor=0.5):
    """Where does `row` fall short of the good rows? Among the numeric columns `blame` found real (never
    the `frozen` ones), take the `most` on which the row is furthest from the median of the `good`
    rows, on the good side of its own value, at least `floor` standard deviations. Move each to that
    median. Returns ([(column, from, to)], changed row), or None if the row falls short nowhere."""
    spec, cells, changes = averages(t), list(row.cells), []
    for name, d, real in found:
        c = next(c for c in t.x if c.name == name)
        v = cells[c.at]
        if not real or not c.num or name in frozen or v == "?":
            continue
        med = sorted(r.cells[c.at] for r in good)[len(good) // 2]
        z = (med - v) / spec[c.at][1] * (1 if d < 0 else -1)
        if z >= floor:  # ranked by how far he trails times how much the column separates good from bad
            changes.append((z * abs(d), name, v, med, c.at))
    changes = sorted(changes, reverse=True)[:most]
    for _, _, _, med, at in changes:
        cells[at] = med
    return ([(n, v, m) for _, n, v, m, _ in changes], Row(cells, tuple(c.norm(cells[c.at]) for c in t.x), row.y)) if changes else None


def evaluate(t, rng, pool, ref, found, good, want, frozen, ask, who, things, goal=0, trials=60):
    """Make a plan for `trials` of the `pool` rows, then check each the only honest way available: do
    the real rows in `ref` (never used by the plan) most like the changed row reach `want` on `goal`?"""
    name, at = t.y[goal].name, t.y[goal].at
    done = [(r, improve(t, r, found, good, frozen)) for r in rng.sample(pool, min(trials, len(pool)))]
    done = [(r, p) for r, p in done if p]
    before, after = [near(t, ref, r, goal=goal) for r, _ in done], [near(t, ref, p[1], goal=goal) for _, p in done]
    print(f"ask {ask}: {things} {who}; target: {name} of {want:.0f}+; {len(done)} of {min(trials, len(pool))} fall short on a rating we can move")
    print(f"  similar real {things}' median {name}: {statistics.median(before):.0f} before, {statistics.median(after):.0f} after; "
          f"{sum(a >= want for a in after) / len(done):.0%} reach the target after, {sum(b >= want for b in before) / len(done):.0%} before")
    print("  e.g. " + ", ".join(f"{n} {a:.0f} -> {b:.0f}" for n, a, b in done[0][1][0]) + "\n")
