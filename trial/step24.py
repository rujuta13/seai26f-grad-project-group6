import step4


def stream(train_csv, rows) -> list[bool]:
    """Decide, row by row, whether each new sign-up is worth a welcome offer.

    A sign-up is kept (True) if no training customer and no earlier sign-up
    beats it on every goal at once, that is, if it is undominated. Each
    decision uses only train_csv and the rows before it, so the answer is
    available as each customer arrives.
    """
    return step4.stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which new sign-ups stop looking like
    train_csv, or None if they never do.

    Sign-ups are checked in consecutive blocks of 25. A block is flagged when
    its distance to the training customers differs significantly from how the
    training customers sit relative to each other. The index returned is the
    last row of the flagged block, and each decision uses only train_csv and
    the rows up to that index.
    """
    return step4.drift(train_csv, list(rows))

