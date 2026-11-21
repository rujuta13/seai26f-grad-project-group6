"""step12.py: guess. Estimate a player's goal values from his other numbers
before a scout is sent, with a range that says how confident the guess is.

The estimate and range come from step2.guess: the mean goal value of the
nearest training players, with a range from the 10th to 90th percentile of
those neighbours, widened by a benchmark-noise floor and by how far the
player's numbers lie from the training players. A goal with no usable
training values gets NaN and an unbounded range, meaning no confident guess.

Reading, column splitting and the neighbour guess come from steps 1 and 2.
Only the wrapper that makes sure every goal is reported is new here.
"""
from __future__ import annotations

import csv
import math

from step1 import _columns
from step2 import guess as _neighbour_guess


def _header(path) -> list[str]:
    """Column names of a MOOT csv, read from its first line only."""
    with open(str(path), newline="", encoding="utf-8") as f:
        first = next(csv.reader(f), [])
    return [h.strip() for h in first]


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a player described by row (goal columns removed).

    Returns {goal: (estimate, low, high)} for every goal column in train_csv.
    low..high is the range the true value is expected to fall in. A wider
    range means less confidence. A goal with no usable training values gets
    (NaN, -inf, inf): no estimate and no confidence.
    """
    _, gmax, gmin = _columns(_header(train_csv))
    goals = gmax + gmin
    unknown = (math.nan, -math.inf, math.inf)

    estimates = _neighbour_guess(train_csv, row)
    out: dict[str, tuple[float, float, float]] = {}
    for g in goals:
        est, low, high = estimates.get(g, unknown)
        out[g] = (float(est), float(low), float(high))
    return out

