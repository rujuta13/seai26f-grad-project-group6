"""step23.py: certify whether two sets of numbers really differ.

Answers the client's questions, such as whether the last campaign really lifted
purchases or whether families with teenagers really shop differently, by reusing
the two-sided permutation test in step13 (which wraps step3).
"""

from step13 import same as _same


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise),
    False if the difference is statistically real at the 5% level.

    xs, ys: lists of numbers (e.g. purchase counts before and after a campaign,
    or purchases of families with and without teenagers). Non-finite values are
    ignored. If either sample is empty, the result is True.
    """
    return bool(_same(xs, ys))

