"""step21.py: remember. Group customers into the kinds of shopper they really
are, and find the best customers on all goals together.

families() clusters customers on their input columns (demographics, household,
marital status, education, and the spend columns that are inputs), using the
k-means of step1. Goal columns are never used, so the clusters describe who the
customers are rather than how the goals turned out.

best() is step1.best: the Pareto-ranked rows over all goal columns together.
In this file those goals include the purchase counts across web, catalogue and
store channels (NUMWEBPURCHASES+, NUMCATALOGPURCHASES+, NUMSTOREPURCHASES+),
tenure (CUSTOMER_DAYS+), and complaints (COMPLAIN-, where lower is better).

Reading, column splitting, feature scaling, k-means and the Pareto best set all
come from step1. Only the thin wrappers below are new here.
"""
from __future__ import annotations

from step1 import K_FAMILIES, _columns, _features, _kmeans, _read
from step1 import best as _pareto_best


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Families are k-means clusters over the input columns. Goal columns are not
    used. A file with no rows gives an empty list. A file with rows but no
    input columns puts every row in family 0.
    """
    header, rows = _read(str(train_csv))
    inputs, _, _ = _columns(header)
    if not rows:
        return []
    if not inputs:
        return [0] * len(rows)
    return _kmeans(_features(rows, inputs), K_FAMILIES)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good customers (best on all goals together),
    at most 5% of rows, best first. Delegates to step1.best.
    """
    return _pareto_best(train_csv)

