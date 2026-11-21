"""Step 21: remember. Customer types and the best customers.

Standard library only. Both client questions are answered by step 11:
  - families(): the shopper types, via step 1's flag-family clustering
    (k-means on the input columns).
  - best(): the unusually good customers, the Pareto front on all goals,
    ranked by mean normalised goal score and capped at 5% of rows.

This module exposes step 11's logic under the names the client asked for.
"""

from step11 import families as _families, best as _best


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return list(_families(train_csv))


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    ranked by mean normalised goal value and capped at 5% of rows."""
    return list(_best(train_csv))

