import numpy as np

_ALPHA = 0.05
_RESAMPLES = 10000


def same(xs, ys) -> bool:
    """True if the two samples do NOT differ beyond run-to-run noise, False if the difference is real.

    Two-sided permutation test on the difference of means, at the 5% level.
    Empty samples, or samples whose means are identical, count as not different.
    """
    x = np.asarray(list(xs), dtype=float).ravel()
    y = np.asarray(list(ys), dtype=float).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return True

    observed = abs(float(x.mean()) - float(y.mean()))
    if observed == 0.0:
        return True

    n = len(x)
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(0)
    threshold = observed - 1e-12 * max(1.0, observed)

    count = 0
    for _ in range(_RESAMPLES):
        p = rng.permutation(pooled)
        d = abs(float(p[:n].mean()) - float(p[n:].mean()))
        if d >= threshold:
            count += 1

    p_value = (count + 1) / (_RESAMPLES + 1)
    return p_value >= _ALPHA

