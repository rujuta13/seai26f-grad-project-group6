"""step29.py - client question 29: spot.

odd(train_csv, rows) gives one suspicion score per row (goals included).
Higher means more likely a data-entry mistake (such as an impossible age or
an odd income), a glitch, or a customer unlike anything in train_csv. Around
1 is typical for training-like data.

The work is done by step9.odd, which combines two parts: how far the flag set
lies from every training flag set, and how far each goal deviates from what
the nearest training neighbours give. Both parts are calibrated on train_csv.
This module only materialises the rows and returns plain floats.

Reused from earlier steps:
  step9: odd (flag distance plus goal deviation, calibrated on train_csv)
"""
from step9 import odd as _odd

__all__ = ["odd"]


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row (goals included). Around 1 is typical for
    training-like data; larger means more likely a glitch, a typo, or unlike
    anything in train_csv."""
    rows = list(rows)
    return [float(s) for s in _odd(train_csv, rows)]

