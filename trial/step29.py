"""step29.py: flag customer records that look wrong (impossible ages, odd incomes, typos),
and customers who behave unlike anyone else.

The work is done by step19.odd, which wraps step9.odd. Each row gets a suspicion score
combining goal-side and input-side isolation, and rows with missing or non-numeric goals
score infinity. Fraud is not decided here: a high score means "check this record", not
"this customer is fraudulent".
"""

from step19 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more likely
    a glitch, a typo, or unlike anything in train_csv.
    """
    return [float(s) for s in _odd(train_csv, rows)]

