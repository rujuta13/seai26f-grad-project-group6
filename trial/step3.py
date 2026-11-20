"""step3.py: certify whether two sets of measurements really differ.

same(xs, ys) is True when the gap between the two samples is within benchmark
noise, and False when the difference is real. Parsing and quantiles come from
step1 and step2; only the bootstrap comparison is new here.
"""
from __future__ import annotations

import math
import random

from step1 import _SEED
from step2 import NOISE_FLOOR, _quantile

BOOTSTRAPS = 2000      # resamples used to estimate how much the gap can wobble
CONFIDENCE = 0.95      # the gap must stay away from zero at this level


def _finite(xs) -> list[float]:
    """Return the usable numbers in xs as floats, dropping anything else."""
    out = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def same(xs, ys) -> bool:
    """True if the samples xs and ys are NOT really different (the gap is noise).

    Two checks must both show a real difference before this returns False:
      1. The gap in means is larger than NOISE_FLOOR times the baseline mean.
      2. A seeded bootstrap interval for the gap at CONFIDENCE excludes zero.
    With fewer than two usable values in either sample there is no evidence of
    a difference, so the result is True.
    """
    a = _finite(xs)
    b = _finite(ys)
    if len(a) < 2 or len(b) < 2:
        return True

    ma = _mean(a)
    mb = _mean(b)
    gap = mb - ma

    # Practical check: a gap within the noise floor of the baseline is noise.
    if abs(gap) <= NOISE_FLOOR * abs(ma):
        return True

    # Statistical check: bootstrap the gap in means and see if zero is plausible.
    rng = random.Random(_SEED)
    na = len(a)
    nb = len(b)
    diffs = []
    for _ in range(BOOTSTRAPS):
        sa = sum(rng.choices(a, k=na)) / na
        sb = sum(rng.choices(b, k=nb)) / nb
        diffs.append(sb - sa)

    alpha = (1.0 - CONFIDENCE) / 2.0
    low = _quantile(diffs, alpha)
    high = _quantile(diffs, 1.0 - alpha)
    return low <= 0.0 <= high

