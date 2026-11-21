"""Step 24: flow control for new customer sign-ups.

stream() decides which incoming sign-ups are worth a welcome offer: a row is
kept when no training customer and no earlier sign-up Pareto-dominates it on
the goals. drift() returns the index of the first sign-up at which recent
sign-ups stop looking like the training customers, or None if they never do.
Both delegate to step4, which holds the implementations.
"""

import step4


def stream(train_csv, rows) -> list[bool]:
    """Return one keep flag per incoming row, judged only against train_csv and earlier rows."""
    return [bool(k) for k in step4.stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Return the index into rows of the first drift warning, or None if there is none."""
    idx = step4.drift(train_csv, rows)
    return None if idx is None else int(idx)

