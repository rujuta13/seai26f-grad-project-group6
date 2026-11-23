"""step23.py: certify whether two sets of customer measurements really differ.

Both client questions reduce to comparing two samples of numbers: purchases (or
response) for customers who were and were not in the last campaign, and purchases
for families with and without teenagers. step3 already implements that check as a
two-sided permutation test on the difference of means at the 5% level, so this
module reuses it rather than rewriting it.
"""
from __future__ import annotations

from step3 import same as _permutation_same


def same(xs, ys) -> bool:
    """True if the two samples do NOT differ beyond noise, False if the gap is real.

    xs, ys: lists of numbers (e.g. purchase counts of two groups of customers).
    """
    return _permutation_same(xs, ys)

