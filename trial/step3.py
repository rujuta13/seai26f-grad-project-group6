import math

import numpy as np


def _clean(values):
    """Convert to a 1-D float array, dropping missing and non-finite entries."""
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def _average_ranks(values):
    """Ranks 1..N with ties replaced by their average rank."""
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Rank range for each unique value is [start+1, start+count]; average is midpoint.
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[inverse]


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise), False otherwise.

    Uses a two-sided permutation test on the difference in mean rank (a
    Mann-Whitney-style test). It makes no distributional assumption, so it is
    robust to skewed run times and outliers. The test is deterministic (fixed
    seed). The significance level is 0.05.
    """
    a = _clean(xs)
    b = _clean(ys)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        # No evidence of any difference.
        return True

    pooled = np.concatenate([a, b])
    if np.all(pooled == pooled[0]):
        # Identical values everywhere: no difference.
        return True

    ranks = _average_ranks(pooled)
    N = n + m
    observed = abs(ranks[:n].mean() - ranks[n:].mean())

    rng = np.random.default_rng(0)
    total = 10000
    chunk = 1000
    extreme = 0
    done = 0
    tol = 1e-12
    while done < total:
        size = min(chunk, total - done)
        perm = np.argsort(rng.random((size, N)), axis=1)
        R = ranks[perm]
        diffs = np.abs(R[:, :n].mean(axis=1) - R[:, n:].mean(axis=1))
        extreme += int(np.sum(diffs >= observed - tol))
        done += size

    p_value = (extreme + 1) / (total + 1)
    return bool(p_value >= 0.05)

