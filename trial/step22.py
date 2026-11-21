"""step22.py: guess a new customer's web and store purchases, and whether they
will answer the next campaign.

The client's two questions are the goals NUMWEBPURCHASES+, NUMSTOREPURCHASES+
and RESPONSE+. The estimates and ranges come from step12.guess, the calibrated
nearest-neighbour model, so every goal in the file is returned in the same
format.

Purchase counts cannot be negative and RESPONSE is a 0/1 flag, so each range is
clipped to the span of values seen in the training file. Clipping preserves
low <= estimate <= high because it is a monotone map.

Reuses step1 (loading and column roles) and step12 (guessing and calibration).
Standard library only.
"""
import math
import os

from step1 import _load, _columns
from step12 import guess as _guess

_BOUNDS = {}  # (path, mtime, size) -> {goal: (min, max)} for the latest file


def _bounds(train_csv):
    """{goal: (smallest, largest)} of the finite training values of each goal."""
    key = (os.path.abspath(train_csv),
           os.path.getmtime(train_csv),
           os.path.getsize(train_csv))
    b = _BOUNDS.get(key)
    if b is None:
        cols, rows = _load(train_csv)
        _, maxi, mini = _columns(cols, rows)
        b = {}
        for g in maxi + mini:
            vals = [r[g] for r in rows
                    if isinstance(r[g], float) and math.isfinite(r[g])]
            if vals:
                b[g] = (min(vals), max(vals))
        _BOUNDS.clear()
        _BOUNDS[key] = b
    return b


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal column of train_csv.

    `row` is a customer's flags with the goal columns removed. For
    NUMWEBPURCHASES+ and NUMSTOREPURCHASES+ the estimate is the expected number
    of purchases through that channel. For RESPONSE+ it is the chance of
    answering the campaign. low..high is the range the true value is fairly
    sure to fall in, clipped to the values seen in training.
    """
    pred = _guess(train_csv, row)
    bounds = _bounds(train_csv)
    out = {}
    for g, (est, lo, hi) in pred.items():
        if g in bounds:
            gmin, gmax = bounds[g]
            est = min(max(est, gmin), gmax)
            lo = min(max(lo, gmin), gmax)
            hi = min(max(hi, gmin), gmax)
        out[g] = (est, lo, hi)
    return out

