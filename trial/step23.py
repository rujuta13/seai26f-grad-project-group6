"""Step 23: certify. Is a difference between two groups real, or just noise?

Standard library only. Reuses the two-sided permutation test from step3, which
compares the difference of means with random relabellings of the pooled values
and needs no normality assumption. Serves both client questions:
  - did the last campaign really lift purchases, or is that noise?
  - do families with teenagers really shop differently, or does it just look
    that way?
"""

from step3 import same as _same


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise (they are not really
    different), False if the difference is real."""
    return _same(xs, ys)

