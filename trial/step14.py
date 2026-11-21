"""step14.py: streaming keep decisions and drift warnings for new scouting rows.

stream: a row is kept when no training row or earlier row dominates it, so the
kept rows form the running Pareto front of the goals (step4's rule). Rows with
missing or non-finite goals are never kept. Non-finite values are excluded
because NaN comparisons make dominance checks unreliable.

drift: each row's goals are checked against the calibrated nearest-neighbour
range from step12.guess. Rows with no usable goals are skipped, so they do not
dilute the signal. The fraction of goals outside the range is averaged over the
last WINDOW scored rows. Training-like rows sit near 0.1, because step12's
ranges cover about 90% of leave-one-out errors. A mean above DRIFT_LEVEL
signals that new rows no longer look like the training rows.

Standard library only.
"""
import math
from collections import deque

from step1 import _load
from step4 import _add, _goals
from step12 import guess

WINDOW = 8          # number of recent scored rows in the drift window
DRIFT_LEVEL = 0.3   # mean out-of-range fraction that triggers a warning


def _orient(row, goals):
    """Goal vector of a row, oriented so higher is better; None if any goal
    is missing, non-numeric or not finite."""
    vec = []
    for c, sgn in goals:
        try:
            x = float(row[c])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        vec.append(sgn * x)
    return vec


def stream(train_csv, rows) -> list[bool]:
    """For each row in order, True if it is worth keeping (good on the goals).

    A row is kept when no training row and no earlier kept or seen row is at
    least as good on every goal and better on at least one.
    """
    cols, train = _load(train_csv)
    goals = _goals(cols, train)
    if not goals:
        return [False] * len(rows)

    front = []
    for r in train:
        v = _orient(r, goals)
        if v is not None:
            _add(front, v)

    keep = []
    for row in rows:
        v = _orient(row, goals)
        keep.append(v is not None and _add(front, v))
    return keep


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the stream stops looking like
    train_csv, or None if it never does."""
    cols, train = _load(train_csv)
    goals = [c for c, _ in _goals(cols, train)]
    if not goals:
        return None

    recent = deque(maxlen=WINDOW)
    for i, row in enumerate(rows):
        pred = guess(train_csv, row)
        counted = out = 0
        for g in goals:
            v = row.get(g)
            if not isinstance(v, float) or not math.isfinite(v):
                continue
            counted += 1
            _, lo, hi = pred[g]
            if v < lo or v > hi:
                out += 1
        if counted == 0:
            continue
        recent.append(out / counted)
        if len(recent) == WINDOW and sum(recent) / WINDOW > DRIFT_LEVEL:
            return i
    return None

