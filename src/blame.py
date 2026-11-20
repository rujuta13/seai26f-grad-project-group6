"""Blame: which input columns separate the bad rows from the good ones."""
from stats import same


def split(t, rows, goal, share=0.1):
    """(worst, best) `share` of `rows` on one goal, whichever way that goal points."""
    i = t.y.index(goal)
    rows = sorted(rows, key=lambda r: abs(goal.heaven - r.y[i]))
    k = max(1, int(share * len(rows)))
    return rows[-k:], rows[:k]


def blame(t, bad, good, rng, n=100):
    """Per x column: (name, how much bigger it is in bad than in good, real?), biggest first.
    A flag's size is how often it is on; a number's is its normalised (0 to 1) value.
    `real` is the step-3 `same` test on `n` sampled rows per side."""
    out = []
    for i, c in enumerate(t.x):
        size = (lambda r: r.x[i]) if c.num else (lambda r: float(r.cells[c.at] == "1"))
        a, b = ([size(r) for r in rng.sample(g, min(n, len(g))) if size(r) != "?"] for g in (bad, good))
        out.append((c.name, sum(a) / len(a) - sum(b) / len(b), not same(sorted(a), sorted(b))))
    return sorted(out, key=lambda o: -abs(o[1]))


def show(t, found, bad, good, label, top=6):
    """Print the `top` columns blame found real with the two groups' means (label = their two names)."""
    mean = lambda g, c: sum(r.cells[c.at] for r in g) / len(g)
    at = {c.name: c for c in t.x}
    for name, d, real in [f for f in found if f[2]][:top]:
        print(f"  {name:22} {label[0]} {mean(bad, at[name]):5.1f}   {label[1]} {mean(good, at[name]):5.1f}")
