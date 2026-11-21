"""Step 19: flag player records that look like data-entry mistakes.

odd() gives one suspicion score per row (higher means more likely a glitch,
a typo, or unlike anything in train_csv). It delegates to step9.odd, which
scores each row's goal values against the most similar training rows and adds
a novelty term for unusual flag sets. Scores are relative, so compare them
with each other rather than against a fixed cut-off.
"""

from step9 import odd as _odd_step9


def odd(train_csv, rows) -> list[float]:
    """Return one suspicion score per row of `rows` (higher = more likely a glitch or unusual).

    Rows carry their goal columns, so a goal value far from what similar
    training builds achieved raises the score, as does a flag set unlike any
    in train_csv.
    """
    return [float(s) for s in _odd_step9(train_csv, rows)]

