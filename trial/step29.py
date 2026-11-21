"""step29.py: spot, for the two client questions.

Q1: "Flag customer records that look wrong: impossible ages, odd incomes, typos."
Q2: "Which customers behave unlike anyone else? Are they fraud, or just unusual?"

Both questions are answered by the suspicion score in step9.odd, which step19.odd
already wraps so that every score is a finite float. A high score means the flag
set lies far from every training flag set (unusual inputs, such as an impossible
age or an odd income), or the goal values disagree with training customers who
have the same flags (a likely typo or glitch). The score does not prove fraud.
A high score only says the record deserves a manual check.

Nothing new is computed here, so this module reuses step19.odd and only fixes
the return type.
"""
from __future__ import annotations

from step19 import odd as _spot


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included). Higher means more
    likely a glitch, a typo, or a customer unlike anyone in train_csv.

    Each score is a finite, non-negative float. Scores of 0.0 mean nothing
    unusual was found, or that there were too few training rows to compare
    against. A high score is a reason to check the record, not proof of fraud.
    """
    return [float(s) for s in _spot(train_csv, rows)]

