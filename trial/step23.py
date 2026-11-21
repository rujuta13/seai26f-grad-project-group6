"""Step 23: certify whether two samples really differ.

same(xs, ys) runs a two-sided permutation test on the difference of means.
It pools the values, repeatedly shuffles them into two groups of the original
sizes, and counts how often a random split gives a gap at least as large as
the observed one. If that happens often, the gap is consistent with chance and
same() returns True. If it is rare (p < 0.05), the difference is treated as
real and same() returns False. The shuffles use a fixed seed, so the answer is
reproducible.
"""
import math
import random

_ALPHA = 0.05
_N_PERM = 5000
_SEED = 1


def _clean(vals):
    """Return the finite numeric values of vals as floats, skipping the rest."""
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def same(xs, ys) -> bool:
    """True if xs and ys do not differ significantly (the gap is noise).

    False if the difference in means is unlikely to be chance at the 5% level.
    Empty samples give True, since no real difference can be shown.
    """
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        return True

    n1, n2 = len(a), len(b)
    n = n1 + n2
    pooled = a + b
    total = sum(pooled)
    sum_a = sum(a)
    observed = abs(sum_a / n1 - (total - sum_a) / n2)

    # Tolerance for floating-point ties between observed and permuted gaps.
    scale = max(1.0, max(abs(v) for v in pooled))
    eps = 1e-9 * scale

    rng = random.Random(_SEED)
    idx = list(range(n))
    hits = 0
    for _ in range(_N_PERM):
        rng.shuffle(idx)
        s = sum(pooled[i] for i in idx[:n1])
        d = abs(s / n1 - (total - s) / n2)
        if d >= observed - eps:
            hits += 1

    # Add-one correction keeps the p-value strictly positive and valid.
    p = (hits + 1) / (_N_PERM + 1)
    return p >= _ALPHA

