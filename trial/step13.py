"""Step 13: certify. Decide whether a gap between two groups is real or noise.

same(xs, ys) runs a two-sided two-sample permutation test on the difference of
means. If the two groups were really alike, shuffling the pooled values between
the groups would often produce a gap at least as large as the observed one.
Both client questions use this check: whether the shortlist differs from the
league average, and whether strikers differ from midfielders on penalty taking.

Standard library only. A fixed seed keeps results reproducible. Non-numeric and
non-finite values are dropped before testing.
"""
import math
import random

ALPHA = 0.05         # below this p-value, the gap is treated as real
PERMUTATIONS = 4000  # number of label shuffles in the permutation test
SEED = 1             # fixed seed so results are reproducible


def _clean(values):
    """Floats from values, leaving out anything that is not a finite number."""
    out = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise, False if the difference is real."""
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        return True  # nothing to compare, so no evidence of a difference

    na = len(a)
    nb = len(b)
    pooled = a + b
    total = sum(pooled)

    observed = abs(sum(a) / na - sum(b) / nb)
    if observed == 0.0:
        return True

    tol = 1e-12 * max(1.0, observed)
    rng = random.Random(SEED)
    count = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        sa = sum(pooled[:na])
        ma = sa / na
        mb = (total - sa) / nb
        if abs(ma - mb) >= observed - tol:
            count += 1

    # Add-one correction keeps the p-value away from exactly zero.
    p = (count + 1) / (PERMUTATIONS + 1)
    return p >= ALPHA

