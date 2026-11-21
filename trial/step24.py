"""step24.py: welcome offers for new sign-ups, and a warning when sign-ups drift.

stream(train_csv, rows): keep a new sign-up when no training row and no earlier
row is at least as good on every goal and better on one. This is step14.stream,
reused unchanged.

drift(train_csv, rows): the index of the first new sign-up at which sign-ups stop
looking like train_csv, judged against the calibrated nearest-neighbour ranges
of step12.guess. This is step14.drift, reused unchanged.

Standard library only.
"""
from step14 import stream as _stream, drift as _drift


def stream(train_csv, rows) -> list[bool]:
    """For each row in order, True if it is worth a welcome offer (good on the goals).

    Decisions use only train_csv and the rows before each one.
    """
    return _stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which new sign-ups stop looking like
    train_csv, or None if they never do."""
    return _drift(train_csv, list(rows))

