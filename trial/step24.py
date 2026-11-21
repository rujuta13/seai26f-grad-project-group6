"""step24.py - client question 24: flow.

stream(train_csv, rows) keeps each new row that is not dominated on all goals
together by any training row or by any earlier row of the stream. Only
train_csv and the rows already seen are used.

drift(train_csv, rows) returns the index of the first row at which the recent
window of rows has stopped looking like train_csv in flag space, or None if it
never does.

Both behaviours are already implemented in step4 (stream and drift) and match
this specification, so they are reused here rather than rewritten.
"""
from step4 import drift as _drift
from step4 import stream as _stream

__all__ = ["stream", "drift"]


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each row that is not dominated on all goals together by any
    training row or by any earlier row of this stream. Uses only train_csv and
    the rows before it."""
    return [bool(k) for k in _stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the recent window stops
    looking like train_csv, or None if it never does."""
    i = _drift(train_csv, rows)
    return None if i is None else int(i)

