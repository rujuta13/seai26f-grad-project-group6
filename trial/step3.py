"""step3.py - client question 4: is a difference real or just noise?

same(xs, ys) decides whether two lists of measurements (e.g. run times)
differ by more than run-to-run noise.

It runs a two-sided permutation test on the difference of means. If the
two groups came from the same distribution, shuffling the group labels
would often produce a gap at least as large as the observed one. When
that p-value is at least ALPHA, the gap is called noise. The test makes no
distributional assumption, so it suits skewed run-time data.

SEED is reused from step1 so results are reproducible.
"""
import random

from step1 import SEED

ALPHA = 0.05
PERMUTATIONS = 2000


def same(xs, ys):
    """True if the gap between xs and ys is noise, False if it is real."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if not xs or not ys:
        return True  # no evidence of any difference

    nx, ny = len(xs), len(ys)
    total = sum(xs) + sum(ys)
    observed = abs(sum(xs) / nx - sum(ys) / ny)
    if observed == 0:
        return True

    rng = random.Random(SEED)
    pool = xs + ys
    hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        s = sum(pool[:nx])
        gap = abs(s / nx - (total - s) / ny)
        if gap >= observed:
            hits += 1

    p_value = (hits + 1) / (PERMUTATIONS + 1)
    return p_value >= ALPHA

