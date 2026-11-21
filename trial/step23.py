"""step23.py: certify the two customer comparisons.

  1. Did the last campaign really lift purchases, or is that noise?
  2. Do families with teenagers really shop differently, or does it just look
     that way?

Both questions compare two groups of purchase numbers. Pass the two lists to
same(): it returns True when the gap is noise, and False only when the
difference is both statistically significant and practically large. The test
is step13.same, which cleans the inputs (missing and non-finite values are
dropped) and then calls step3.same. Standard library only.
"""
from step13 import same as _certify


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise),
    False if the difference is real.

    xs, ys: lists of numbers, e.g. purchase counts of two groups of customers.
    Missing and non-finite entries are ignored. If either group is left empty,
    there is no evidence of a difference, so the result is True.
    """
    return bool(_certify(xs, ys))

