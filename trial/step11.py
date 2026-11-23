"""Step 11: client-facing wrappers around the step1 functions.

families(train_csv): sorts players into types with KMeans, one label per row.
best(train_csv): row indices of the unusually good builds, judged on all goals
together, at most 5% of rows.

Note for the client: the second request ("strong and good from the spot") names
only two goals, Strength+ and Penalties+. best() is specified to rank on all
goals together, so it does not answer that narrower question. A subset-of-goals
version would need its own function with a goal list parameter.
"""

from step1 import best as _best
from step1 import families as _families


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return [int(x) for x in _families(train_csv)]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows."""
    return [int(i) for i in _best(train_csv)]

