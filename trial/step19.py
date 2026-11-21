"""step19.py - client question 19: spot.

odd(train_csv, rows) gives one suspicion score per row (goals included).
Higher means more likely a data-entry mistake, a glitch, or a player unlike
anything in train_csv. Around 1 is typical for training-like data.

Reused from earlier steps:
  step9: odd (flag distance plus goal deviation, calibrated on train_csv)
"""
from step9 import odd as _odd

__all__ = ["odd"]


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row (goals included). Around 1 is typical for
    training-like data; larger means more likely a glitch, a typo, or unlike
    anything in train_csv."""
    return [float(s) for s in _odd(train_csv, rows)]

