"""step12.py: scout-facing guesses of each goal for an unseen player, with confidence ranges.

The model and the range come from step2.guess. This wrapper keeps the output
in the form the client asked for: plain floats, with low <= estimate <= high.
"""

import math

from step2 import guess as _guess_base


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal of train_csv.

    `row` has its goal columns removed. low..high is the range the true value
    is expected to fall in (5th to 95th percentile of out-of-bag errors, added
    around the estimate).
    """
    raw = _guess_base(train_csv, row)
    out: dict[str, tuple[float, float, float]] = {}
    for name, (est, lo, hi) in raw.items():
        est = float(est)
        lo = float(lo)
        hi = float(hi)
        if math.isnan(est) or math.isnan(lo) or math.isnan(hi):
            out[name] = (est, lo, hi)
            continue
        lo, hi = min(lo, est), max(hi, est)
        out[name] = (est, lo, hi)
    return out

