from step4 import drift as _drift
from step4 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Decide keep (True) or not (False) for each row, in arrival order.

    A row is kept if no training row or earlier streamed row dominates it on
    the standardized goals, and its average standardized score is in the top
    5% of the reference rows. Each decision uses only train_csv and the rows
    before it.
    """
    return _stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row where new rows stop looking like train_csv.

    Compares the nearest-training-row distances of the last 20 rows with the
    training set's own leave-one-out nearest-neighbour distances, using the
    permutation test from step3. Returns None if no warning is raised.
    """
    return _drift(train_csv, list(rows))

