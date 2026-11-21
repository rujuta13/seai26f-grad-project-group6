"""step22.py - client question 22: guess.

guess(train_csv, row) estimates every goal of a customer (flag set) that has
not been seen, with a range the true value should fall in. The client's two
questions map onto goals in the training file:
  - online versus in-store purchases: NUMWEBPURCHASES+ and NUMSTOREPURCHASES+;
  - answering the next campaign: RESPONSE+.

The work is done by step12.guess, which gives a median estimate from the
nearest training flag sets and a range calibrated on leave-one-out errors.
This module only reshapes its output so every value is a plain float.

Reused from earlier steps:
  step12: guess (estimate and calibrated low..high range for every goal)
"""
from step12 import guess as _guess

__all__ = ["guess"]


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of a customer and a range the true value should
    fall in. `row` maps input column names to values (goal columns removed).
    Returns {goal: (estimate, low, high)} for every goal column."""
    result = _guess(train_csv, row)
    return {goal: (float(est), float(low), float(high))
            for goal, (est, low, high) in result.items()}

