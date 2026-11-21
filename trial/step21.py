"""step21.py: customer segments and the best-customer front.

families(train_csv): step1's k-means family labels, reused unchanged.

best(train_csv): the same rule as step1.best (the Pareto front of the goals,
capped at BEST_FRACTION of the rows, ranked by normalised goal sum). step11's
fast front is used because step1.best compares every row with every other row,
which is too slow for 2,200 customers. The result is the same as step1.best.

Standard library only.
"""
import step1 as _step1
import step11 as _step11


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return _step1.families(train_csv)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows."""
    return _step11.best(train_csv)

