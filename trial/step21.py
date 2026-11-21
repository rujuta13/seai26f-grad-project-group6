"""step21.py - client question 21: remember.

families(train_csv) groups the customer rows (flag sets) into kinds of
shopper. The work is done by step11.families, which clusters a sample of at
most SAMPLE_SIZE rows with k-means and picks the number of families by
silhouette score, then assigns every row to its nearest centroid.

best(train_csv) returns the unusually good customers: rows not dominated on
all goals together, at most 5% of rows. The work is done by step11.best,
which reuses step1.best.

Both behaviours already exist in step11 and match this specification, so
they are reused here rather than rewritten.
"""
from step11 import best as _best
from step11 import families as _families

__all__ = ["families", "best"]


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return [int(f) for f in _families(train_csv)]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows, in ascending order."""
    return [int(i) for i in _best(train_csv)]

