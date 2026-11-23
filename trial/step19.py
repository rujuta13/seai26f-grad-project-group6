"""Step 19: spot likely data-entry mistakes and unusual player records."""

import math

from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv.

    Scores combine input-space isolation and novelty, unseen categories, and
    goal values that sit far from the training median (robustly scaled), with a
    fixed penalty for missing or non-numeric goal values. Scores are relative
    to the training data and are a prompt to check a record, not a verdict.
    """
    scores = _odd(train_csv, rows)
    out = []
    for s in scores:
        v = float(s)
        if not math.isfinite(v):
            # Should not happen; treat an undefined score as maximally suspicious.
            v = float("inf")
        out.append(v)
    return out

