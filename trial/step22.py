"""Step 22: guess. Two client questions, answered with calibrated ranges.

Standard library only. The estimator, the calibrated band and the check that
every input column is present are reused from step12, which uses step2 and
step1. The only new work is clipping: each estimate and band is limited to the
range that goal takes in the training data. A purchase count therefore cannot
go below zero, and a campaign response (RESPONSE+) stays between 0 and 1.

The goal columns of the answer are NUMWEBPURCHASES+ (online purchases) and
NUMSTOREPURCHASES+ (in-store purchases) for the first question, and RESPONSE+
(answered the campaign) for the second. Every goal in the file is returned.
"""

from step2 import _model
from step12 import guess as _guess


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a new customer.

    `row` holds the input columns only (goal columns removed). Returns
    {goal: (estimate, low, high)} for every goal column of train_csv, where
    low..high is the range the true value should fall in. Estimate and range
    are clipped to the minimum and maximum the goal takes in train_csv.
    Raises ValueError if an input column of train_csv is absent from row.
    """
    result = _guess(train_csv, row)
    model = _model(train_csv)
    targets = model["targets"]
    if not targets:
        return result

    out = {}
    for g, (name, _) in enumerate(model["goals"]):
        lo = min(t[g] for t in targets)
        hi = max(t[g] for t in targets)
        est, low, high = result[name]
        est = min(max(est, lo), hi)
        low = min(max(low, lo), hi)
        high = min(max(high, lo), hi)
        out[name] = (est, low, high)
    return out

