"""step22.py: guesses for a new customer's online and store purchases, and campaign response.

The client's two questions are answered by the goal guesses already in step12, which
wraps step2.guess. Every goal of train_csv is predicted, including NUMWEBPURCHASES+,
NUMSTOREPURCHASES+ and RESPONSE+. Each range is the 5th to 95th percentile of
out-of-bag errors, added around the estimate.
"""

import math

from step12 import guess as _guess_wrapped


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal of train_csv.

    `row` has its goal columns removed. low..high is the range the true value is
    fairly sure to fall in. Estimates are plain floats with low <= estimate <= high.
    """
    raw = _guess_wrapped(train_csv, row)
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

