"""step14.py: streaming keep/drop for scouting reports, and drift warnings for new seasons.

Both client questions are answered by the streaming logic already in step4:
- stream: keep each incoming build that no earlier build (training or incoming) beats on every goal.
- drift: warn at the end of the first block of new rows whose goal values differ from the training goals.

Caveat: drift runs a separate 5% permutation test for every goal in every block, with no
multiple-comparison correction, so over a long season a false alarm becomes more likely.
"""

from step4 import drift as _drift
from step4 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each incoming row that is not dominated on the goals by any row seen so far.

    Uses only train_csv and the rows that have already arrived.
    """
    return _stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Index into `rows` of the first row at which new rows stop looking like train_csv.

    The warning is issued at the last row of the first block that differs, which is the
    earliest point at which that block can be judged. Returns None if no block differs.
    """
    return _drift(train_csv, list(rows))

