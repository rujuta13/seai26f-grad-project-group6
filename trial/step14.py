from step4 import drift as _drift
from step4 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Decide, one row at a time, whether each arriving build is worth following up.

    A row is kept (True) iff no earlier reference build, either from train_csv or
    from the rows already streamed, Pareto-dominates it on the goal columns.
    Each decision uses only train_csv and the rows before it.
    """
    return _stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Return the index into `rows` where new builds stop resembling train_csv.

    Warns at the first row index at which the most recent window of rows is, as
    a group, far from the training inputs (nearest-neighbour distance above the
    training baseline more often than chance allows). Returns None if no such
    warning is raised.
    """
    return _drift(train_csv, list(rows))

