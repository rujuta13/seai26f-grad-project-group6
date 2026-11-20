"""step3.py: certify whether two sets of measurements really differ.

Standard library only (Python 3.13). Nothing from steps 1 or 2 is needed:
this step works on plain lists of numbers, not on MOOT csv rows.

A difference counts as real only if both hold:
  1. Effect size: Cliff's delta is non-negligible (|delta| >= 0.147).
  2. Significance: an Efron bootstrap test on the difference in means gives
     p < 0.05.
Otherwise the gap is treated as run-to-run noise.
"""
from __future__ import annotations

import bisect
import math
import random

_SEED = 1
_BOOTS = 1000
_ALPHA = 0.05
_DELTA_SMALL = 0.147  # Romano et al.: |Cliff's delta| below this is negligible


def _clean(xs) -> list[float]:
    """Numbers as floats, dropping None, NaN and unconvertible cells."""
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out.append(v)
    return out


def _cliff_delta(xs: list[float], ys: list[float]) -> float:
    """Cliff's delta in [-1, 1]: P(x > y) - P(x < y) over all pairs."""
    ys_sorted = sorted(ys)
    m = len(ys_sorted)
    greater = 0  # pairs with x > y
    less = 0     # pairs with x < y
    for x in xs:
        greater += bisect.bisect_left(ys_sorted, x)
        less += m - bisect.bisect_right(ys_sorted, x)
    return (greater - less) / (len(xs) * m)


def _bootstrap_p(xs: list[float], ys: list[float],
                 rng: random.Random) -> float:
    """Efron's two-sample bootstrap p-value for a difference in means.

    Both samples are shifted to the pooled mean, so resampling simulates
    the null hypothesis of no difference. The p-value is the fraction of
    resampled |mean difference| values at least as large as the observed one.
    """
    n, m = len(xs), len(ys)
    grand = sum(xs + ys) / (n + m)
    mx = sum(xs) / n
    my = sum(ys) / m
    observed = abs(mx - my)

    xs0 = [x - mx + grand for x in xs]
    ys0 = [y - my + grand for y in ys]

    hits = 0
    for _ in range(_BOOTS):
        bx = sum(xs0[rng.randrange(n)] for _ in range(n)) / n
        by = sum(ys0[rng.randrange(m)] for _ in range(m)) / m
        if abs(bx - by) >= observed:
            hits += 1
    return (hits + 1) / (_BOOTS + 1)


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise).

    xs, ys: lists of numbers (e.g. run times). Returns False only when the
    difference is both non-negligible in effect size and statistically
    significant. Empty input or identical values count as the same.
    """
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        return True
    if min(a + b) == max(a + b):
        return True
    if abs(_cliff_delta(a, b)) < _DELTA_SMALL:
        return True
    rng = random.Random(_SEED)
    return _bootstrap_p(a, b, rng) >= _ALPHA

