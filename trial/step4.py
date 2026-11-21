"""step4.py - client question 5: flow of nightly benchmark results.

stream(train_csv, rows) keeps a row (True) when no reference point is better
on every goal. The reference points are the training rows plus the rows that
came before it in this stream. Only train_csv and earlier rows are used.

drift(train_csv, rows) warns at the first row where the recent window of rows
has stopped looking like train_csv in flag space. A new row is "out of place"
if its nearest training flag set (standardised as in step2) is farther away
than the OUT_QUANTILE of the training rows' own leave-one-out nearest-neighbour
distances. The warning fires when at least OUT_FRACTION of the last WINDOW rows
are out of place. Under the training distribution that many out-of-place rows
in one window is very unlikely, so ordinary variation does not trigger it.

Helpers are reused from step1 (_load, _sqdist, _dominates) and step2
(_model, _quantile).
"""
import collections
import functools
import math

from step1 import _dominates, _load, _sqdist
from step2 import _model, _quantile

WINDOW = 20
OUT_QUANTILE = 0.99
OUT_FRACTION = 0.5


@functools.lru_cache(maxsize=None)
def _train_goals(train_csv):
    """Goal columns and training goal vectors, maximisation-signed."""
    _, goal_cols, _, Y = _load(train_csv)
    return goal_cols, [list(y) for y in Y]


def _goal_vector(row, goal_cols):
    """Maximisation-signed goal vector of one row."""
    return [sign * float(row[name]) for name, sign in goal_cols]


def stream(train_csv: str, rows) -> list[bool]:
    """Keep (True) each row that is not dominated on all goals by any
    training row or by any earlier row of this stream."""
    goal_cols, train_Y = _train_goals(train_csv)
    reference = list(train_Y)
    keep = []
    for row in rows:
        y = _goal_vector(row, goal_cols)
        dominated = any(_dominates(r, y) for r in reference)
        keep.append(not dominated)
        reference.append(y)
    return keep


@functools.lru_cache(maxsize=None)
def _threshold(train_csv):
    """OUT_QUANTILE of the leave-one-out nearest-neighbour squared distances
    among training flag sets, or None if there are fewer than two rows."""
    _, _, _, _, Z, _ = _model(train_csv)
    n = len(Z)
    if n < 2:
        return None
    nn = []
    for i in range(n):
        nn.append(min(_sqdist(Z[i], Z[j]) for j in range(n) if j != i))
    nn.sort()
    return _quantile(nn, OUT_QUANTILE)


def _nearest_sq(z, Z):
    return min(_sqdist(z, zi) for zi in Z)


def drift(train_csv: str, rows) -> int | None:
    """Index into rows of the first row at which the recent window stops
    looking like train_csv, or None if it never does."""
    input_cols, _, means, stds, Z, _ = _model(train_csv)
    thr = _threshold(train_csv)
    if thr is None or not Z:
        return None

    need = math.ceil(OUT_FRACTION * WINDOW)
    window = collections.deque(maxlen=WINDOW)
    count = 0
    for i, row in enumerate(rows):
        z = [(float(row[c]) - means[t]) / stds[t]
             for t, c in enumerate(input_cols)]
        out = _nearest_sq(z, Z) > thr
        if len(window) == WINDOW:
            count -= window[0]
        window.append(out)
        count += out
        if len(window) == WINDOW and count >= need:
            return i
    return None

