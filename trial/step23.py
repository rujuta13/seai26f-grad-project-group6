import math
import random

_ALPHA = 0.05   # significance level: below this, the gap is treated as real
_PERMS = 2000   # Monte Carlo permutations (p-value resolution ~ 1/2001)
_SEED = 0       # fixed seed so the same inputs always give the same answer


def _clean(v):
    """Keep only finite numbers; skip anything that cannot be read as a float."""
    out = []
    for x in v:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def same(xs, ys) -> bool:
    """True if the two samples do NOT differ significantly (the gap is noise).

    Used for both client questions: whether a campaign lifted purchases, and
    whether families with teenagers shop differently. Two-sided permutation
    test on the difference of means. Distribution-free, so it does not assume
    normal values. Non-numeric and non-finite entries are ignored. Deterministic
    via a fixed seed.
    """
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        # nothing to compare, so no evidence of a real difference
        return True

    na, nb = len(a), len(b)
    ma = sum(a) / na
    mb = sum(b) / nb
    obs = abs(ma - mb)

    # tolerance so floating-point round-off does not hide ties
    scale = max(1.0, abs(ma), abs(mb))
    tol = 1e-12 * scale
    if obs <= tol:
        return True

    pooled = a + b
    rng = random.Random(_SEED)
    hits = 0
    for _ in range(_PERMS):
        rng.shuffle(pooled)
        d = abs(sum(pooled[:na]) / na - sum(pooled[na:]) / nb)
        if d >= obs - tol:
            hits += 1

    # add-one correction keeps the p-value away from exactly zero
    p = (hits + 1) / (_PERMS + 1)
    return p >= _ALPHA

