"""step13.py: certify whether two sets of measurements really differ.

The client asks two questions: whether the shortlist really beats the league
average, and whether strikers really take better penalties than midfielders.
Both reduce to comparing two samples of numbers, so this module exposes the
same() check that step3 already implements (a two-sided permutation test on
the difference of means, at the 5% level) and reuses it rather than
rewriting it.
"""
from __future__ import annotations

from step3 import same as _permutation_same


def same(xs, ys) -> bool:
    """True if the two samples do NOT differ beyond noise, False if the gap is real.

    xs, ys: lists of numbers (e.g. goal values of two groups of players).
    """
    return _permutation_same(xs, ys)

