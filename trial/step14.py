"""Step 14: client-facing flow for incoming scouting reports.

stream(train_csv, rows): one keep/skip decision per incoming row, using only
train_csv and the rows that came before it. A row is kept when no earlier
reference row beats it on every goal.

drift(train_csv, rows): index of the first incoming row at which the recent
rows look unlike train_csv (unusual inputs or goal values, tested over a
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

