"""step24.py: flow, for the two client questions.

Q1: "As new customers sign up, tell me straight away who's worth a welcome offer."
Q2: "Warn me when this month's sign-ups stop looking like our usual customers."

Q1 is answered by step14.stream, the online Pareto test: a new sign-up is kept
when no training customer and no earlier sign-up in the stream is at least as
good on every goal and strictly better on one. Q2 is answered by step14.drift,
which warns at the first row where the recent sign-ups are mostly unlike the
training customers, using the reference limit of step4.drift.

Both decisions come from the flow code in step4 through step14, so this module
reuses them rather than repeating them. It only fixes the return types.
"""
from __future__ import annotations

from step14 import drift as _drift
from step14 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each arriving sign-up that is worth a welcome offer, or drop
    (False) it.

    rows arrive one at a time, in order. Each decision uses only train_csv and
    the rows before it. A row is kept unless a training row or an earlier row
    of the stream is at least as good on every goal and strictly better on one.
    """
    return [bool(keep) for keep in _stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the recent sign-ups have stopped
    looking like train_csv, or None if they never do.

    A warning is raised once at least MIN_UNLIKE of the last WINDOW rows are
    unlike the training input rows. The rule is the one in step4.drift.
    """
    idx = _drift(train_csv, rows)
    return None if idx is None else int(idx)

