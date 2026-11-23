"""Step 24: welcome-offer stream for new sign-ups and a drift warning for each month's batch."""

from step4 import stream as _stream
from step14 import drift as _drift


def stream(train_csv, rows) -> list[bool]:
    """Keep a new sign-up iff no training customer and no earlier kept sign-up
    is at least as good on every goal and strictly better on one.

    Each decision uses only train_csv and the rows before it, so a welcome offer
    can be made as each sign-up arrives.
    """
    rows = list(rows)
    return _stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index into `rows` of the first row at which this month's sign-ups stop
    looking like train_csv, or None if they never do.

    Compares running windows of goal values and input features against the
    training data at geometrically growing checkpoints (16, 32, 64, ...), with
    the false-alarm budget split across checkpoints and features.
    """
    rows = list(rows)
    return _drift(train_csv, rows)

