from step3 import same as _same


def same(xs, ys) -> bool:
    """True if the two samples are NOT really different (the gap is noise), False if the difference is real.

    xs, ys: lists of numbers, e.g. purchase amounts before and after a campaign,
    or purchases of families with and without teenagers. Uses the two-sided
    permutation test on the difference of means at the 5% level from step3.
    """
    return _same(xs, ys)

