"""Step 21: client-facing customer segmentation and best-customer lookup.

families(train_csv): sorts customers into kinds of shopper with KMeans, one
label per row.
best(train_csv): row indices of the unusually good customers, judged on all
goals together, at most 5% of rows.

Note for the client: best() ranks customers on all goals together. The
request for "busy across channels, loyal, no complaints" names a subset of
goals and a complaint condition, which this function does not select on. A
subset-of-goals version would need its own function with a goal list parameter.
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

