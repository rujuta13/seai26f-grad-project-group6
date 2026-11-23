"""step24.py: flow tools for customer sign-ups arriving one at a time.

stream() keeps a new customer when no earlier customer (training or already streamed)
is at least as good on every goal and strictly better on one. The customer goals are
the purchase channels, web visits, complaints, campaign response and tenure, so a kept
customer is one worth a welcome offer.

drift() warns when the last _WINDOW new customers sit significantly further from the
training customers than training customers sit from each other.

Both are reused from step14, which wraps step4 for flow data. The customer file has no
identifier columns (num, Name, url), so step14's identifier filter removes nothing and
the behaviour matches step4 on every trait column.
"""
from __future__ import annotations

from step14 import drift as _drift
from step14 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or discard (False) each new customer, in arrival order.

    A customer is kept if no training customer and no earlier streamed customer is at
    least as good on every goal and strictly better on at least one.
    """
    return _stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index of the first new customer at which the sign-ups stop resembling train_csv.

    Returns None if no warning is raised.
    """
    return _drift(train_csv, rows)

