"""step21.py: customer families and the best customers, at about 2,200 rows.

Both client questions use the same definitions as step1 and step11. step11 is
used because it stays workable at this scale: the silhouette score is estimated
on a fixed sample above 2,000 rows, and the Pareto front is computed in blocks.
"""

from step11 import best as _best_full
from step11 import families as _families_full


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return [int(x) for x in _families_full(train_csv)]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows. Uses the Pareto front over all goals, ranked when the
    front is larger than the cap."""
    return sorted(int(i) for i in _best_full(train_csv))

