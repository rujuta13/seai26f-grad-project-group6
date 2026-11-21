"""step19.py: spot data-entry mistakes and unusual player records.

Standard library only (Python 3.13). Reuses step9, which already implements
this: each row gets a suspicion score from input novelty, unseen input cells,
and goal anomalies (missing or non-finite goals, goals outside the training
range, and goals far from the kNN estimate from step2).
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

