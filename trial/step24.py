"""Step 24: flow. Welcome offers for new customers, and drift warnings.

Standard library only. Both client questions are answered by the streaming
logic of step 4 (exposed as step 14): stream() keeps each new customer who is
not beaten on every goal by a training customer or an earlier arrival, and
drift() warns at the end of the first block of DRIFT_BLOCK sign-ups whose
nearest-neighbour distances differ from the training reference.
"""

from step14 import stream as _stream, drift as _drift


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each new customer worth a welcome offer, one row at a time.

    Each decision uses only train_csv and the rows before it, so the rows may
    arrive one at a time and in order."""
    return _stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index into rows at which the warning first fires, or None if the new
    rows never stop looking like train_csv."""
    return _drift(train_csv, rows)

