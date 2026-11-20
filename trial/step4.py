"""step4.py: streaming keep/drop decisions and drift warnings for nightly runs.

stream: a row is kept when no known row (training or previously seen) dominates
it, so kept rows form the running Pareto front of the goals.

drift: each incoming row's goals are compared with the k-nearest-neighbour
prediction from step2. Over a sliding window, if the goals fall outside the
predicted 10th..90th percentile range too often, new rows no longer look like
the training rows and a warning is issued.

Standard library only.
"""
from collections import deque

from step1 import _load, _columns, _dominates
from step2 import guess

WINDOW = 8          # number of recent rows in the drift window
DRIFT_LEVEL = 0.5   # mean out-of-range fraction that triggers a warning


def _goals(cols, rows):
    """(column, sign) pairs for every goal; sign orients higher-is-better."""
    _, maxi, mini = _columns(cols, rows)
    return [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]


def _orient(row, goals):
    """Goal vector of a row, oriented so higher is better; None if unusable."""
    try:
        return [sgn * float(row[c]) for c, sgn in goals]
    except (KeyError, TypeError, ValueError):
        return None


def _add(front, v):
    """Insert v into the Pareto front in place.

    Returns True if v is non-dominated (it joins the front and any points it
    dominates leave it), False if some point on the front dominates v.
    Dominance is transitive, so the front alone decides future keep decisions.
    """
    if any(_dominates(f, v) for f in front):
        return False
    front[:] = [f for f in front if not _dominates(v, f)]
    front.append(v)
    return True


def stream(train_csv, rows) -> list[bool]:
    """For each row in order, True if it is worth keeping.

    A row is worth keeping when no training row or earlier row is at least as
    good on every goal and better on one. Rows with missing or non-numeric
    goals are not kept.
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
    train_csv, or None if it never does.

    Each row's goals are checked against the step2 prediction from its flags.
    The fraction of goals outside the predicted 10th..90th percentile range is
    averaged over the last WINDOW rows. Training-like rows give a mean near
    0.2, so a mean above DRIFT_LEVEL signals a change.
    """
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
            if not isinstance(v, float):
                continue
            counted += 1
            _, lo, hi = pred[g]
            if v < lo or v > hi:
                out += 1
        recent.append(out / counted if counted else 0.0)
        if len(recent) == WINDOW and sum(recent) / WINDOW > DRIFT_LEVEL:
            return i
    return None

