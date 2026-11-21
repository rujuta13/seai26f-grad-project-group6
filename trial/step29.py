"""Step 29: spot customer records that look wrong or unlike the rest.

odd() gives one suspicion score per row (higher means more likely a glitch,
a typo, or a customer unlike anyone in train_csv). Each row's goal values are
checked against the most similar training customers, and an unusual flag set
adds to the score. It delegates to step9.odd, which holds the implementation.
Scores are relative, so compare them with each other rather than against a
fixed cut-off.
"""

from step9 import odd as _odd_step9


def odd(train_csv, rows) -> list[float]:
    """Return one suspicion score per row of `rows` (higher = more likely a glitch or unusual).

    Rows carry their goal columns, so a goal value far from what similar
    training customers achieved raises the score, as does a flag set unlike
    any in train_csv.
    """
    return [float(s) for s in _odd_step9(train_csv, list(rows))]

