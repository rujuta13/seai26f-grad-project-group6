"""step22.py: guess a new customer's online and in-store purchases and campaign response.

The client's two questions are answered by step12.guess, which predicts every goal
from the customer's other columns. The estimate is the median of the nearest
training customers, and low..high is their 10th to 90th percentile. For the
purchase counts this is the expected number of purchases; for RESPONSE+ it is the
share of similar customers who answered the campaign, rounded to the median.

Identifier filtering (num, Name, url) comes from step11 through step12. The
customer file has no such columns, so every trait column is used.
"""
from __future__ import annotations

from step12 import guess as _guess


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of `row` as (estimate, low, high).

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns {goal column name: (estimate, low, high)} for every goal in the
    training file, in header order. Relevant goals include NUMWEBPURCHASES+,
    NUMSTOREPURCHASES+ and RESPONSE+.
    """
    return dict(_guess(train_csv, row))

