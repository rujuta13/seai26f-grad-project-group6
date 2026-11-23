from step1 import best as _best
from step11 import families as _families_std


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Clusters on all input flags after standardizing each column, so large-scale
    columns such as INCOME do not dominate the grouping.
    """
    return _families_std(train_csv)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together), at most 5% of rows.

    Uses the Pareto-front ranking from step1: a row qualifies if no other row is
    at least as good on every goal and strictly better on one.
    """
    return _best(train_csv)

