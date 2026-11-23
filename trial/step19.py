"""Step 19: client-facing suspicion scores for player records.

odd(train_csv, rows): one suspicion score per row (goals included). Higher
scores mean the record is more likely a data-entry mistake or is unlike
anything in train_csv, so it is worth a second look before trusting its
ratings.
"""

from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    rows = list(rows)
    scores = [float(s) for s in _odd(train_csv, rows)]
    if len(scores) != len(rows):
        raise ValueError("odd() returned a score count that does not match rows")
    return scores

