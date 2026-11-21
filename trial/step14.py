"""Step 14: flow control for scouting reports and season drift.

stream() keeps each incoming report that is not Pareto-dominated on the goals
by the training rows or by the reports that arrived before it. drift() returns
the index of the first incoming row at which new rows have stopped looking like
the training rows, or None if they never do. Both delegate to step4, which
holds the implementations.
"""

import step4


def stream(train_csv, rows) -> list[bool]:
    """Return one keep flag per incoming row, judged only against train_csv and earlier rows."""
    return [bool(k) for k in step4.stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Return the index into rows of the first drift warning, or None if there is none."""
    idx = step4.drift(train_csv, rows)
    return None if idx is None else int(idx)

