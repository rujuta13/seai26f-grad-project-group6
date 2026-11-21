"""Step 21: remember. Customer segments and best-customer rows for the client.

Serves the client's questions:
  - "Group our 2,200 customers into the kinds of shopper they really are."
  - "Who are our best customers: busy across channels, loyal, no complaints?"

Thin wrappers around step11, which already handles customer-sized files:
families() clusters customers on their input columns, and best() ranks them on
all goals together (Pareto layers, at most 5% of rows). Results are remembered
per file, so repeated client requests return the stored answer.
"""

import step11


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Customers are clustered on their input (flag and behaviour) columns with
    k-means, k-means++ seeding and several restarts; the lowest-SSE solution is
    kept. Each customer gets the label of the nearest cluster centre.
    """
    return step11.families(train_csv)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good customers, best on all goals together.

    Each goal is min-max normalised so that higher is better (minimised goals
    are negated; missing values count as worst). Customers are taken from
    successive Pareto fronts, best front first, ordered within a front by total
    normalised score. At most 5% of rows are returned, best first.
    """
    return step11.best(train_csv)

