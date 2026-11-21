import numpy as np


_ALPHA = 0.05
_N_PERM = 10000
_SEED = 0


def same(xs, ys) -> bool:
    """Return True if the two samples do not differ significantly (gap is noise).

    Uses a two-sided permutation test on the difference in means, so it makes
    no distributional assumptions (run times are often skewed). Non-finite
    values are dropped. With fewer than two usable values in either sample,
    no real difference can be established, so True is returned.
    """
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return True

    observed = abs(float(x.mean()) - float(y.mean()))
    if observed == 0.0:
        return True

    pooled = np.concatenate([x, y])
    n_x = x.size
    total = pooled.sum()
    rng = np.random.default_rng(_SEED)

    count = 0
    for _ in range(_N_PERM):
        idx = rng.permutation(pooled.size)[:n_x]
        sx = pooled[idx].sum()
        mean_x = sx / n_x
        mean_y = (total - sx) / (pooled.size - n_x)
        if abs(mean_x - mean_y) >= observed - 1e-12:
            count += 1

    p_value = (count + 1) / (_N_PERM + 1)
    return bool(p_value >= _ALPHA)

