"""step3.py: decide whether two sets of measurements differ for real.

Used to answer questions like "is turning on `inline` really faster, or is it
run-to-run noise?". The test is a two-sided permutation test on the difference
of means, so it makes no normality assumption and works with small samples.
"""

import numpy as np

_ALPHA = 0.05
_N_PERM = 10000
_SEED = 0


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise),
    False if the difference is statistically real at the 5% level.

    Non-finite values are ignored. If either sample is empty, there is no
    evidence of a difference, so the result is True.
    """
    x = np.asarray([float(v) for v in xs], dtype=float)
    y = np.asarray([float(v) for v in ys], dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return True

    n_x = x.size
    pooled = np.concatenate([x, y])
    observed = abs(x.mean() - y.mean())
    if observed == 0.0:
        return True

    # Under the null hypothesis the labels are exchangeable, so reshuffle the
    # pooled values and count how often a gap at least as large appears.
    rng = np.random.default_rng(_SEED)
    tol = 1e-12 * max(1.0, observed)
    hits = 0
    for _ in range(_N_PERM):
        perm = rng.permutation(pooled)
        gap = abs(perm[:n_x].mean() - perm[n_x:].mean())
        if gap >= observed - tol:
            hits += 1

    # Add-one correction keeps the p-value away from an impossible exact zero.
    p_value = (hits + 1) / (_N_PERM + 1)
    return p_value >= _ALPHA

