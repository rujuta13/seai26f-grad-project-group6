"""Step 13: certify whether two groups of measurements really differ.

same(xs, ys) runs a two-sided permutation test on the difference of means.
It answers both client questions: whether the shortlist's goal values beat the
league average, and whether strikers and midfielders differ as penalty takers.
It uses only the standard library and a fixed random seed, so repeated calls
give the same answer. Non-numeric and non-finite values are ignored.
"""

import math
import random


def same(xs, ys) -> bool:
    """Return True if the gap between xs and ys is consistent with noise.

    Returns False only when the difference in means is statistically
    significant at the 5% level (two-sided permutation test). Returns True
    when either sample has no usable values, or when the means are equal.
    """
    alpha = 0.05
    n_perm = 10000

    groups = []
    for seq in (xs, ys):
        vals = []
        for x in seq:
            try:
                f = float(x)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                vals.append(f)
        groups.append(vals)
    a, b = groups

    if not a or not b:
        return True

    na = len(a)
    nb = len(b)
    observed = abs(math.fsum(a) / na - math.fsum(b) / nb)
    if observed == 0.0:
        return True

    pooled = a + b
    rng = random.Random(0)
    threshold = observed * (1.0 - 1e-9)
    hits = 0

    for _ in range(n_perm):
        rng.shuffle(pooled)
        sa = math.fsum(pooled[:na])
        sb = math.fsum(pooled[na:])
        if abs(sa / na - sb / nb) >= threshold:
            hits += 1

    p_value = (hits + 1) / (n_perm + 1)
    return bool(p_value > alpha)

