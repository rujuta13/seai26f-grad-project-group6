"""step14.py: flow. Decide keep/drop for each arriving scouting report, and
warn when the new players stop looking like the training players.

Both questions are answered by the flow code in step4, so this module reuses
it rather than repeating it. stream() keeps a row when no training row and no
earlier row of the stream dominates it, which is the online form of the Pareto
test in step1.best. drift() warns at the first row where the recent rows are
mostly unlike the training rows, using the same reference limit as step4.
"""
from __future__ import annotations

from step4 import drift as _drift
from step4 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each arriving row that is good on the goals, or drop (False).

    rows arrive one at a time, in order. Each decision uses only train_csv and
    the rows before it. A row is kept unless a training row or an earlier row
    of the stream is at least as good on every goal and strictly better on one.
    """
    return [bool(keep) for keep in _stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the recent rows have stopped
    looking like train_csv, or None if they never do.

    A warning is raised once at least MIN_UNLIKE of the last WINDOW rows are
    unlike the training input rows. The rule is the one in step4.drift.
    """
    return _drift(train_csv, rows)

