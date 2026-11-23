from step14 import drift as _drift
from step14 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Decide keep (True) or not (False) for each row as it arrives.

    A row is worth a welcome offer if no training row or earlier streamed row
    beats it on every standardized goal, and its average standardized score is
    in the top 5% of the reference rows. Each decision uses only train_csv and
    the rows before it.
    """
    return _stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which new sign-ups stop looking like train_csv.

    Compares the nearest-training-row distances of the last 20 rows with the
    training set's own leave-one-out nearest-neighbour distances, using the
    permutation test from step3. Returns None if the rows never drift.
    """
    return _drift(train_csv, list(rows))

