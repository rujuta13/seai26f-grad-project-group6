"""step21.py: customer segments and the best-customer shortlist.

families() groups customers into kinds of shopper. step1.families already does this:
it clusters the input columns (standardised, KMeans, with k chosen by silhouette),
so the segmentation uses only customer traits and not the goal columns.

best() picks the best customers: the Pareto front on all goals (channel purchases,
web visits, complaints, response and tenure), capped at 5% of rows. step1.best already
does this, so the client's question "busy across channels, loyal, no complaints" maps
directly onto the goal columns of the file.

Both functions delegate to step1 rather than repeating its logic.
"""
from __future__ import annotations

import step1


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return step1.families(train_csv)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together), at most 5% of rows.

    Returned indices are sorted.
    """
    return step1.best(train_csv)

