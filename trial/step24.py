"""step24.py: flow over new customer sign-ups.

Standard library only (Python 3.13). Reuses step14, which already implements
both jobs on top of step4:
  - stream(): keeps a new sign-up only if it is a new best on at least one
    goal, so a kept row is worth a welcome offer. Each decision uses only
    train_csv and the rows before it.
  - drift(): compares blocks of new rows against the training reference with
    step3.same and returns the index of the last row in the first block that
    differs. A final partial block is not tested.
"""
from __future__ import annotations

from step14 import drift as _step14_drift
from step14 import stream as _step14_stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or drop (False) each arriving sign-up, in arrival order.

    A row is kept when no training row and no earlier kept row is at least
    as good on every goal, so a kept row is a new best on at least one goal.
    Each decision uses only train_csv and the rows before it.
    """
    return [bool(keep) for keep in _step14_stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first warning that new sign-ups have stopped
    looking like train_csv, or None if they never do.

    Rows are tested in blocks of 20. The returned index is the last row of
    the first block whose input distances differ from the training reference.
    """
    return _step14_drift(train_csv, rows)

