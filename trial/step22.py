"""step22.py: guess a new customer's online and store purchases, and whether
they will answer the next campaign.

Standard library only (Python 3.13). Reuses step12.guess, which already gives
every goal a kNN median estimate and a calibrated 80% band. For this file the
goals are NUMWEBPURCHASES+, NUMCATALOGPURCHASES+, NUMSTOREPURCHASES+,
NUMDEALSPURCHASES+, NUMWEBVISITSMONTH+, RESPONSE+ and CUSTOMER_DAYS+. The two
client questions read from that dict: NUMWEBPURCHASES+ and NUMSTOREPURCHASES+
for purchases, and RESPONSE+ for the campaign.
"""
from __future__ import annotations

from step12 import guess as _step12_guess


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of a new customer, with a range that should hold
    the true value about 80% of the time.

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns {goal: (estimate, low, high)} for every goal column. For example,
    the online-versus-store question reads NUMWEBPURCHASES+ and
    NUMSTOREPURCHASES+, and the campaign question reads RESPONSE+ (0 or 1).
    A goal with no usable neighbour values gets (nan, nan, nan).
    """
    return _step12_guess(train_csv, row)

