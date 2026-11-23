from step3 import same as _same


def same(xs, ys) -> bool:
    """True if the two samples do NOT differ beyond run-to-run noise, False if the difference is real.

    xs, ys: lists of numbers (e.g. run times). Uses the two-sided permutation
    test at the 5% level from step3.
    """
    return _same(xs, ys)

