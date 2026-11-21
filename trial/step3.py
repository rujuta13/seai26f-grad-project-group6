"""Step 3: decide whether two sets of measurements really differ.

same(xs, ys) runs a two-sided permutation test on the difference of means.
It uses only the standard library and a fixed random seed, so repeated calls
give the same answer. Non-numeric and non-finite values are ignored.
"""

import math
import random


def same(xs, ys) -> bool:
    """Return True if the gap between xs and ys is consistent with noise.

    Returns False only when the difference in means is statistically
    significant at the 5% level (two-sided permutation test).
    Returns True when either sample has no usable values.
    """
    alpha = 0.05
    n_perm = 10000

    a = []
    for x in xs:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            a.append(f)

    b = []
    for y in ys:
        try:
            f = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            b.append(f)

    if not a or not b:
        return True

    na = len(a)
    nb = len(b)
    total = math.fsum(a) + math.fsum(b)

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
        sb = total - sa
        diff = abs(sa / na - sb / nb)
        if diff >= threshold:
            hits += 1

    p_value = (hits + 1) / (n_perm + 1)
    return bool(p_value > alpha)

