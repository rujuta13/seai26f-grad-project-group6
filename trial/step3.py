import random

ALPHA = 0.05        # below this p-value, the gap is treated as real
PERMUTATIONS = 4000 # number of label shuffles in the permutation test
SEED = 1            # fixed seed so results are reproducible


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise, False if the difference is real.

    Two-sample permutation test on the difference of means. If the groups
    were not really different, shuffling the pooled values between the two
    groups would often produce a gap at least as large as the observed one.
    """
    a = [float(x) for x in xs]
    b = [float(y) for y in ys]
    if not a or not b:
        return True  # nothing to compare, so no evidence of a difference

    na = len(a)
    pooled = a + b
    n = len(pooled)
    nb = n - na

    observed = abs(sum(a) / na - sum(b) / nb)
    if observed == 0.0:
        return True

    rng = random.Random(SEED)
    count = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        ma = sum(pooled[:na]) / na
        mb = sum(pooled[na:]) / nb
        if abs(ma - mb) >= observed - 1e-12:
            count += 1

    # Add-one correction keeps the p-value away from exactly zero.
    p = (count + 1) / (PERMUTATIONS + 1)
    return p >= ALPHA

