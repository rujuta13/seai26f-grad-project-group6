"""Blame: which input columns separate the bad rows from the good ones."""
from stats import same


def split(t, rows, goal, share=0.1):
    """(worst, best) `share` of `rows` on one goal (lower is better)."""
    rows = sorted(rows, key=lambda r: float(r.cells[goal.at]))
    k = max(1, int(share * len(rows)))
    return rows[-k:], rows[:k]


def blame(t, bad, good, rng, n=100):
    """Per x column: (name, how much likelier it is on in bad than in good, real?), biggest first.
    `real` is the step-3 `same` test on `n` sampled rows per side."""
    out = []
    for c in t.x:
        a, b = ([float(r.cells[c.at] == "1") for r in rng.sample(g, min(n, len(g)))] for g in (bad, good))
        out.append((c.name, sum(a) / len(a) - sum(b) / len(b), not same(sorted(a), sorted(b))))
    return sorted(out, key=lambda o: -abs(o[1]))
