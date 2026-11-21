"""step13.py: certify the client's two comparisons.

  1. Is the shortlist really better than the league average, or is that luck?
  2. Are strikers really better penalty takers than midfielders?

Both questions compare two groups of goal values. Pass the two lists of values
to same(): it returns True when the gap is noise, and False only when the
difference is both statistically significant and practically large. The test
itself is step3.same, reused unchanged. This module only cleans the inputs:
missing, non-numeric and non-finite values (e.g. NaN from empty player cells)
are dropped first, so they cannot corrupt the medians or the effect size.
Standard library only.
"""
import math

from step3 import same as _step3_same


def _clean(values):
    """The finite float values of an iterable; anything else is dropped."""
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise),
    False if the difference is real.

    xs, ys: lists of numbers (e.g. run times, or goal values of two groups).
    Missing and non-finite entries are ignored. If either group is left empty,
    there is no evidence of a difference, so the result is True.
    """
    return bool(_step3_same(_clean(xs), _clean(ys)))

