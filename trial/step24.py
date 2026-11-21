"""Step 24: flow for new customer sign-ups.

Two questions from the client:

1. "As new customers sign up, tell me straight away who's worth a welcome
   offer." stream(train_csv, rows) decides each sign-up as it arrives, using
   only the training customers and the sign-ups before it. A sign-up is kept
   when no known customer is at least as good on every goal.

2. "Warn me when this month's sign-ups stop looking like our usual customers."
   drift(train_csv, rows) returns the index of the first sign-up after which
   the new rows no longer match the training data, or None if they never do.

Both checks are the sequential procedures from step14, so they carry the same
guarantees: a row is judged only against earlier information, and the
false-warning chance over the whole stream is at most 0.05.
"""
from step14 import drift as _drift14
from step14 import stream as _stream14


def stream(train_csv, rows) -> list[bool]:
    """Decide keep (True) or not (False) for each row, in arrival order.

    A row is kept when no training customer and no earlier sign-up is at least
    as good on every goal. Goals ending in '+' are maximised and goals ending
    in '-' are minimised. Rows with a missing or non-numeric goal value are not
    kept.
    """
    return _stream14(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first sign-up at which new rows stop looking like train_csv.

    Returns None if no warning is raised.
    """
    return _drift14(train_csv, rows)

