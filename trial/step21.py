"""Step 21: the client's two customer requests as named entry points.

families() groups customers into types by k-means clustering over the input
(flag) columns only. best() returns the row indices of the customers who are
not Pareto-dominated on the goals, capped at 5% of rows. Both delegate to
step1, which holds the implementations.
"""

import step1


def families(train_csv) -> list[int]:
    """Return one family (cluster) label per data row of train_csv, in file order."""
    return [int(f) for f in step1.families(train_csv)]


def best(train_csv) -> list[int]:
    """Return sorted row indices of the best customers on all goals together, at most 5% of rows."""
    return [int(i) for i in step1.best(train_csv)]

