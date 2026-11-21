"""Step 3: certify whether two sets of measurements really differ.

Standard library only. The seed is reused from step1 so results are
reproducible.

Method: a two-sided permutation test on the difference of means. It makes
no normality assumption, which suits run times. The p-value is the share
of random relabellings of the pooled data whose mean gap is at least as
large as the observed gap (with the usual +1 correction).
"""

import random

from step1 import SEED


ALPHA = 0.05          # significance level: p <= ALPHA means a real difference
PERMUTATIONS = 4000   # random relabellings used to estimate the p-value


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise, False if it is a real difference."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    nx, ny = len(xs), len(ys)
    if nx == 0 or ny == 0:
        # Nothing to compare, so no difference can be claimed.
        return True

    observed = abs(sum(xs) / nx - sum(ys) / ny)
    if observed == 0.0:
        return True

    pooled = xs + ys
    total = sum(pooled)
    rng = random.Random(SEED)
    tol = observed * (1.0 - 1e-9)  # guard against floating-point ties

    extreme = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        sa = sum(pooled[:nx])
        gap = abs(sa / nx - (total - sa) / ny)
        if gap >= tol:
            extreme += 1

    p_value = (extreme + 1) / (PERMUTATIONS + 1)
    return p_value > ALPHA

