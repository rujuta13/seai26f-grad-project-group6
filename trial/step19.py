"""step19.py: flag suspicious player records and unusual flag sets for the spot check.

odd(train_csv, rows): one suspicion score per row, goals included. Higher means
more likely a data-entry mistake or a record unlike anything in train_csv.

The scoring is step9.odd, reused unchanged. Each score adds two parts: the
distance from the row's flags to the nearest training flag set, in units of
the typical leave-one-out distance inside the training data, and the worst
goal deviation from what the nearest training flag sets achieve, in units of
each goal's spread. A missing or non-finite goal scores step9.MISSING_SCORE.
Standard library only.
"""
from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row of rows (goals included): higher means more
    likely a glitch, a typo, or a record unlike anything in train_csv."""
    return _odd(train_csv, rows)

