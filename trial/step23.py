"""step23.py - client question 23: certify.

same(xs, ys) decides whether two lists of numbers differ by more than noise.
Both client questions reduce to this test:
  - did the last campaign really lift purchases, or is the lift noise?
    (xs and ys are purchase counts for customers with and without the campaign);
  - do families with teenagers really shop differently, or does it only look
    that way? (xs and ys are a shopping measure for the two groups).

The work is done by step13.same, which wraps step3.same: a two-sided
permutation test on the difference of means, with ALPHA, PERMUTATIONS and
SEED taken from step3 and step1.
"""
from step13 import same as _same

__all__ = ["same"]


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise (they are not really
    different), False if the difference is real."""
    return bool(_same(xs, ys))

