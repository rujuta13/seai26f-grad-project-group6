"""step21.py: customer families and the best customers.

Standard library only (Python 3.13). Reuses step11, which already does both
jobs at scale. With about 2,200 customers the file is above step11's sample
size, so families() clusters a fixed sample and assigns every row to its
nearest cluster centre. best() gives the Pareto front of the goal scores,
capped at 5% of rows.
"""
from __future__ import annotations

from step11 import best as _step11_best
from step11 import families as _step11_families


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Labels are contiguous integers starting at 0.
    """
    return _step11_families(train_csv)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together).

    Pareto front over the normalised goals, ordered by mean score, and at
    most 5% of the rows.
    """
    return _step11_best(train_csv)

