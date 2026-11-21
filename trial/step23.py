"""Step 23: certify. Did the last campaign really lift purchases, or is that noise?
Do families with teenagers really shop differently, or does it just look that way?

Method: a two-sided permutation test on the difference of means. It makes no
normality assumption, so it suits skewed spending and the many ties of 0/1
campaign-response and small-count purchase data. A difference counts as real
only when it is both statistically significant (p < _ALPHA) and non-negligible
in size (|Cliff's delta| >= _MIN_DELTA). Otherwise the gap is treated as noise.
"""

import bisect
import math
import random

_ALPHA = 0.05          # significance level for the permutation test
_PERMUTATIONS = 5000   # number of random relabellings
_MIN_DELTA = 0.147     # Cliff's delta threshold for a "small" effect (Romano et al.)
_SEED = 12345          # fixed seed so results are reproducible


def _clean(xs):
    """Return the finite numeric values of xs as floats; other entries are dropped."""
    out = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _cliffs_delta(xs, ys_sorted):
    """Cliff's delta in [-1, 1]: P(x > y) - P(x < y), using a sorted copy of ys."""
    m = len(ys_sorted)
    total = 0
    for x in xs:
        less = bisect.bisect_left(ys_sorted, x)
        greater = m - bisect.bisect_right(ys_sorted, x)
        total += greater - less
    return total / (len(xs) * m)


def _permutation_p(a, b, observed):
    """Two-sided permutation p-value for an absolute difference of means >= observed.

    The statistic is symmetric in the two groups, so the smaller group is the
    one drawn at random from the pooled values. This keeps each relabelling
    cheap even when the other group is very large.
    """
    if len(a) <= len(b):
        small, large = a, b
    else:
        small, large = b, a
    pooled = small + large
    k = len(small)
    n_large = len(large)
    n_all = len(pooled)
    grand = math.fsum(pooled)

    rng = random.Random(_SEED)
    hits = 0
    for _ in range(_PERMUTATIONS):
        picked = rng.sample(range(n_all), k)
        s = math.fsum(pooled[i] for i in picked)
        d = abs(s / k - (grand - s) / n_large)
        if d >= observed or math.isclose(d, observed, rel_tol=1e-9, abs_tol=1e-12):
            hits += 1
    return (hits + 1) / (_PERMUTATIONS + 1)


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise, False if the difference is real.

    xs, ys: lists of numbers (e.g. goal values of customers who were sent a
    campaign and of those who were not, or spending of two family groups).
    Non-numeric and non-finite entries are ignored. With no usable data on
    either side there is no evidence of a difference, so the answer is True.
    """
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        return True

    observed = abs(math.fsum(a) / len(a) - math.fsum(b) / len(b))
    if observed == 0.0:
        return True

    p_value = _permutation_p(a, b, observed)
    if p_value >= _ALPHA:
        return True
    return abs(_cliffs_delta(a, sorted(b))) < _MIN_DELTA

