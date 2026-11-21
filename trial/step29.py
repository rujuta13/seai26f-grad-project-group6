"""step29.py: spot customer records that look wrong, and customers unlike anyone else.

Standard library only (Python 3.13). Reuses step9.odd, which already gives
each row a suspicion score from input novelty, unseen input cells, and goal
anomalies. Scores are not a fraud verdict: a high score means the record is
unlike the training file, which could be a data-entry error or an unusual
but genuine customer. Checking the flagged cells against source records
tells the two apart.
"""
from __future__ import annotations

from step9 import odd as _step9_odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included).

    Higher means more likely a glitch, a typo, or unlike anything in
    train_csv. Scores are finite and non-negative. Each row is scored
    against train_csv on its own, so rows in the same batch do not
    influence each other.
    """
    return [float(s) for s in _step9_odd(train_csv, rows)]

