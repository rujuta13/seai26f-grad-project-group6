"""step29.py: flag suspicious customer records and unusual behaviour.

odd(train_csv, rows): one suspicion score per row, goals included. Higher means
more likely a typo, an impossible value, or a record unlike anything in
train_csv. A high score marks a record for checking. It does not show fraud,
since unusual but genuine customers also score high.

The scoring is step9.odd, reused unchanged. It adds two parts: the distance
from the row's flags to the nearest training flag set, in units of the typical
leave-one-out distance inside the training data, and the worst goal deviation
from what the nearest training flag sets achieve, in units of each goal's
spread. A missing or non-finite goal scores step9.MISSING_SCORE.

Standard library only.
"""
from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row of rows (goals included): higher means more
    likely a glitch, a typo, or a record unlike anything in train_csv."""
    return _odd(train_csv, list(rows))

