"""Certify a gap between two groups of rows: for each goal, real or noise?"""
from statistics import median

from stats import same


def gap(t, a, b):
    """Per goal: (median of a, median of b, whether `same` calls the two groups the same)."""
    out = []
    for c in t.y:
        xs, ys = sorted(r.cells[c.at] for r in a), sorted(r.cells[c.at] for r in b)
        out.append((median(xs), median(ys), same(xs, ys)))
    return out


def say(g, who):
    """One line for one goal's gap: noise, or real and by how much."""
    m1, m2, noise = g
    return "noise" if noise else f"real: {who} is {abs(m1 - m2) / m2:.1%} {'lower' if m1 < m2 else 'higher'}"
