"""Step 29: client-facing suspicion scores for customer records.

odd(train_csv, rows): one suspicion score per row (goals included). Higher
scores mean the record is more likely a data-entry error, a typo, or simply
unlike anything in train_csv. A high score is a reason to check a record, not
evidence of fraud. Unusual but genuine customers also score high.
"""

import math

from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    rows = list(rows)
    scores = [float(s) if math.isfinite(float(s)) else 0.0
              for s in _odd(train_csv, rows)]
    if len(scores) != len(rows):
        raise ValueError("odd() returned a score count that does not match rows")
    return scores

