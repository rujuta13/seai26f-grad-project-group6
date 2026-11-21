import math
import random

_ALPHA = 0.05   # significance level: below this, the gap is treated as real
_PERMS = 2000   # Monte Carlo permutations (p-value resolution ~ 1/2001)
_SEED = 0       # fixed seed so the same inputs always give the same answer


def _clean(v):
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
    """True if the two samples do NOT differ significantly (gap is noise).

    Two-sided permutation test on the difference of means. Distribution-free,
    so it does not assume normal run times. Deterministic via a fixed seed.
    """
    a = _clean(xs)
    b = _clean(ys)
    if not a or not b:
        # nothing to compare, so no evidence of a real difference
        return True

    na, nb = len(a), len(b)
    obs = abs(sum(a) / na - sum(b) / nb)

    # tolerance so floating-point round-off does not hide ties
    scale = max(1.0, abs(sum(a) / na), abs(sum(b) / nb))
    tol = 1e-12 * scale

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

