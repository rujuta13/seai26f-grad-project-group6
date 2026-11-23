"""step19.py: flag player records that look like data-entry mistakes, and players so
unusual their ratings deserve a second look.

The work is done by step9.odd, which scores each row by goal-side and input-side
isolation, with missing or non-numeric goals treated as certain glitches. This
wrapper only makes sure the result has one plain float per row.
"""

from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more likely
    a glitch, a typo, or unlike anything in train_csv.
    """
    rows = list(rows)
    if not rows:
        return []
    scores = [float(s) for s in _odd(train_csv, rows)]
    if len(scores) != len(rows):
        raise RuntimeError(
            f"expected {len(rows)} scores, got {len(scores)}"
        )
    return scores

