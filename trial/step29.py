"""Step 29: flag customer records that look wrong, and customers unlike anyone else."""

import math

from step19 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv.

    The score combines input-space isolation and novelty, unseen categories,
    and goal values far from the training median, plus a penalty for missing
    or non-numeric goal values. It measures how unusual a record is relative
    to train_csv. It does not establish fraud, error, or intent: a high score
    is a reason to look at the record, not a finding about the customer.
    """
    rows = list(rows)
    scores = _odd(train_csv, rows)
    out = []
    for s in scores:
        v = float(s)
        if not math.isfinite(v):
            v = float("inf")
        out.append(v)
    return out

