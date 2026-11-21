"""Step 19: spot. Suspicion scores for player records that may be data-entry mistakes.

Standard library only. The client's two questions (flagging likely data-entry
mistakes, and flagging players whose ratings should be checked before use) are
both answered by step 9's suspicion score, so the scoring is reused unchanged.
This module exposes it under the name the client asked for and checks that
one score comes back per row.
"""

from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    scores = _odd(train_csv, rows)
    if len(scores) != len(rows):
        raise RuntimeError("expected one suspicion score per row")
    return scores

