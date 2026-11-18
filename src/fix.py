"""Fix: the fewest changes that turn a bad row into a good one."""
from guess import guess


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
