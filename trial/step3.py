"""step3.py: are two sets of measurements (e.g. run times) really different?

A difference is called real only if both checks pass:
  1. Significance: a permutation test on the difference of medians gives
     p < ALPHA, so the gap is unlikely to be run-to-run noise.
  2. Practical size: |Cliff's delta| >= SMALL_DELTA, so a tiny but
     significant gap is not reported as a real difference.
Otherwise the gap is treated as noise. Standard library only.
"""
import random
import statistics

ALPHA = 0.05          # significance level for the permutation test
SMALL_DELTA = 0.147   # Romano et al.'s threshold for a negligible effect
N_PERM = 2000         # number of random permutations
SEED = 1              # fixed seed so results are reproducible


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise),
    False if the difference is real."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if not xs or not ys:
        return True  # no evidence of any difference

    # Effect size: Cliff's delta, in [-1, 1].
    gt = lt = 0
    for x in xs:
        for y in ys:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    delta = (gt - lt) / (len(xs) * len(ys))
    if abs(delta) < SMALL_DELTA:
        return True  # negligible effect, whatever the p-value

    # Significance: permutation test on the absolute difference of medians.
    observed = abs(statistics.median(xs) - statistics.median(ys))
    pooled = xs + ys
    nx = len(xs)
    rng = random.Random(SEED)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        a = pooled[:nx]
        b = pooled[nx:]
        if abs(statistics.median(a) - statistics.median(b)) >= observed:
            hits += 1
    p = (hits + 1) / (N_PERM + 1)
    return p >= ALPHA

