import numpy as np

_ALPHA = 0.05
_N_PERM = 10000
_SEED = 0
_CHUNK = 500


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is consistent with noise (not really different).

    Uses a two-sided permutation test on the difference in means. Non-finite
    values are dropped. With an empty sample or an observed difference of
    exactly zero, the samples are treated as not really different.
    """
    x = np.asarray([float(v) for v in xs], dtype=float)
    y = np.asarray([float(v) for v in ys], dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return True

    observed = abs(float(x.mean()) - float(y.mean()))
    if observed == 0.0:
        return True

    pooled = np.concatenate([x, y])
    total = n + m
    rng = np.random.default_rng(_SEED)

    threshold = observed * (1.0 - 1e-12)
    count = 0
    done = 0
    while done < _N_PERM:
        b = min(_CHUNK, _N_PERM - done)
        perm = rng.random((b, total)).argsort(axis=1)
        vals = pooled[perm]
        diff = np.abs(vals[:, :n].mean(axis=1) - vals[:, n:].mean(axis=1))
        count += int(np.count_nonzero(diff >= threshold))
        done += b

    p_value = (count + 1) / (_N_PERM + 1)
    return bool(p_value > _ALPHA)

