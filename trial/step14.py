"""step14.py: flow over scouting reports and season drift.

Standard library only (Python 3.13). Reuses step4, which already implements
this flow: stream() keeps a row only if it is a new best on at least one
goal, using only train_csv and the earlier rows. drift() compares blocks of
new rows against the training reference with step3.same and returns the
index of the last row in the first block that differs.
"""
from __future__ import annotations

from step4 import drift as _drift
from step4 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or drop (False) each arriving row, in arrival order.

    A row is kept when no training row and no earlier kept row is at least
    as good on every goal, so a kept row is a new best on at least one goal.
    Each decision uses only train_csv and the rows before it.
    """
    return _stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index into rows where new rows first stop looking like train_csv.

    Rows are tested in blocks of 20. The returned index is the last row of
    the first block whose input distances differ from the training reference,
    since that is the earliest row at which the difference can be seen.
    Returns None if no complete block differs. A final partial block is not
    tested.
    """
    return _drift(train_csv, rows)

