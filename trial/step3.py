"""step3.py: certify whether two sets of measurements really differ.

Nothing in step1 or step2 computes a significance test, so this module is
self-contained and uses only the standard library and numpy 1.26.

same() runs a two-sided permutation test on the difference of means. Shuffling
the pooled values breaks any real link between a value and its group, so the
p-value is the share of shuffles that produce a gap at least as large as the
observed one. A permutation test makes no normality assumption, which suits
noisy run times.
"""
from __future__ import annotations

import numpy as np

_ALPHA = 0.05       # significance level: gaps with p <= _ALPHA count as real
_N_PERM = 10000     # number of random re-labellings
_CHUNK = 1000       # permutations generated per batch, to bound memory
_SEED = 0           # fixed seed so repeated calls give the same answer


def _as_sample(values, name):
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def same(xs, ys) -> bool:
    """True if the two samples do NOT differ beyond noise, False if the gap is real.

    xs, ys: lists of numbers (e.g. run times). The test is two-sided at the
    5% level on the difference of means.
    """
    x = _as_sample(xs, "xs")
    y = _as_sample(ys, "ys")
    n = x.size
    pooled = np.concatenate([x, y])
    total = pooled.size

    observed = abs(x.mean() - y.mean())
    if observed == 0.0:
        return True

    rng = np.random.default_rng(_SEED)
    exceed = 0
    done = 0
    while done < _N_PERM:
        c = min(_CHUNK, _N_PERM - done)
        perms = rng.permuted(np.tile(pooled, (c, 1)), axis=1)
        diffs = np.abs(perms[:, :n].mean(axis=1) - perms[:, n:].mean(axis=1))
        exceed += int(np.count_nonzero(diffs >= observed - 1e-12))
        done += c

    # Add-one correction keeps the p-value away from an impossible exact zero.
    p_value = (exceed + 1) / (_N_PERM + 1)
    return p_value > _ALPHA

