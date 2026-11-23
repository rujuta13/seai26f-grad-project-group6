"""step13.py: certify whether two sets of numbers really differ.

Answers the client's questions by reusing the two-sided permutation test in step3.
"""

from step3 import same as _same


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise),
    False if the difference is statistically real at the 5% level.

    xs, ys: lists of numbers (e.g. run times). Non-finite values are ignored.
    If either sample is empty, the result is True.
    """
    return bool(_same(xs, ys))

