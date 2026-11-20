"""Step 3: certify whether two sets of measurements really differ.

Method: a two-sided permutation test on the difference of medians. It makes no
normality assumption and is robust to outliers. A difference counts as real only
when it is both statistically significant (p < _ALPHA) and non-negligible in
size (|Cliff's delta| >= _MIN_DELTA). Otherwise the gap is treated as noise.
"""

import math
import random
import statistics

_ALPHA = 0.05          # significance level for the permutation test
_PERMUTATIONS = 2000   # number of random relabellings
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


def _cliffs_delta(xs, ys):
    """Cliff's delta in [-1, 1]: P(x > y) - P(x < y) over all pairs."""
    greater = 0
    less = 0
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (len(xs) * len(ys))


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise, False if the difference is real.

    xs, ys: lists of numbers (e.g. run times). Non-numeric and non-finite
    entries are ignored. With no usable data on either side there is no
    evidence of a difference, so the answer is True.
    """
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        return True

    observed = abs(statistics.median(a) - statistics.median(b))
    if observed == 0.0:
        return True

    na = len(a)
    pooled = a + b
    rng = random.Random(_SEED)
    hits = 0
    for _ in range(_PERMUTATIONS):
        rng.shuffle(pooled)
        d = abs(statistics.median(pooled[:na]) - statistics.median(pooled[na:]))
        if d >= observed or math.isclose(d, observed):
            hits += 1
    p_value = (hits + 1) / (_PERMUTATIONS + 1)

    if p_value >= _ALPHA:
        return True
    return abs(_cliffs_delta(a, b)) < _MIN_DELTA

