"""step13.py - client question 13: certify.

same(xs, ys) decides whether two lists of numbers differ by more than noise.
It reuses step3.same, a two-sided permutation test on the difference of means,
with the ALPHA, PERMUTATIONS and SEED settings defined in step3 and step1.

The client asks two things, and both reduce to this test:
  - whether the shortlist's goal values really beat the league average, or
    whether the gap is luck;
  - whether strikers really take better penalties than midfielders, or whether
    the gap only looks that way.
"""
from step3 import same as _same


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise (they are not really
    different), False if the difference is real."""
    return bool(_same(xs, ys))

