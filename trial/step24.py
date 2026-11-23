"""Step 24: client-facing flow for new customer sign-ups.

stream(train_csv, rows): one keep/skip decision per incoming sign-up, using
only train_csv and the sign-ups that came before it. A sign-up is kept when no
earlier reference row beats it on every goal.

drift(train_csv, rows): index of the first incoming sign-up at which the recent
sign-ups look unlike train_csv (unusual inputs or goal values, tested over a
sliding window), or None if they never do.
"""

from step4 import drift as _drift
from step4 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each incoming row that is on the running Pareto front of
    train_csv plus the rows before it; skip (False) the rest."""
    rows = list(rows)
    return [bool(keep) for keep in _stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Index into `rows` of the first row at which new rows have stopped
    looking like train_csv, or None if they never do."""
    rows = list(rows)
    idx = _drift(train_csv, rows)
    return None if idx is None else int(idx)

