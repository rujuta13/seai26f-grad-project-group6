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
