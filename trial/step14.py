import step4


def stream(train_csv, rows) -> list[bool]:
    """Keep each streamed row that no training row, and no earlier streamed row,
    dominates on the goals. Each decision uses only train_csv and the rows before it."""
    return step4.stream(train_csv, list(rows))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which new rows stop looking like
    train_csv, or None if they never do. Each decision uses only train_csv and
    the rows up to that index."""
    return step4.drift(train_csv, list(rows))

