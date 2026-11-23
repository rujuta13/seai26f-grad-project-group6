"""step24.py: welcome-offer shortlist for new sign-ups, and drift warnings for each month's sign-ups.

Both client questions reuse the streaming logic already in step14 (and step4 underneath):
- stream: keep (True) each new sign-up that no earlier row, from training or from this
  month's arrivals, beats on every goal. Only train_csv and the rows already seen are used.
- drift: warn at the last row of the first block of new rows whose goal values differ from
  the training goals. Returns None if no block differs.
"""

from step14 import drift as _drift
from step14 import stream as _stream


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each incoming sign-up that is not dominated on the goals by any row seen so far."""
    return [bool(k) for k in _stream(train_csv, list(rows))]


def drift(train_csv, rows) -> int | None:
    """Index into `rows` at which new rows first stop looking like train_csv, or None if they never do."""
    result = _drift(train_csv, list(rows))
    return None if result is None else int(result)

