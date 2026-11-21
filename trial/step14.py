"""Step 14: flow. Which scouting reports to follow up, and when new rows drift.

Standard library only. Both client questions are answered by the streaming
logic of step 4, which reuses the dominance test from step 1, the cached
training model from step 2, and the permutation test from step 3. This module
exposes that logic under the names the client asked for.

stream: each row is decided using only train_csv and the rows before it. A
row is kept when no training row and no earlier row of the stream is at least
as good on every goal, with one strictly better (or identical).

drift: the index returned is the last row of the first block of DRIFT_BLOCK
rows whose nearest-neighbour distances differ from the training reference. That
is the first row at which the warning can be issued, because the block must be
complete before the test can run.
"""

from step4 import stream as _stream, drift as _drift


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each row worth following up on, one row at a time.

    Each decision uses only train_csv and the rows before it, so the rows may
    arrive one at a time and in order."""
    return _stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index into rows at which the warning first fires, or None if the rows
    never stop looking like train_csv."""
    return _drift(train_csv, rows)

