"""Step 21: customer families and best customers for the marketing brief."""

from step1 import best as _best
from step1 import families as _families


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return _families(train_csv)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together), at most 5% of rows."""
    return _best(train_csv)

