"""step4.py: flow. Decide keep/drop for each arriving benchmark row, and warn
when arriving rows stop looking like the training rows.

Reading, column splitting, number parsing, distances, quantiles and missing-value
filling come from step1 and step2. The keep rule is the Pareto test from
step1.best, applied to one row at a time against everything seen so far.
"""
from __future__ import annotations

import math
import random
from collections import deque

from step1 import _SEED, _columns, _dist2, _num, _read
from step2 import _fill, _quantile

REFERENCE_ROWS = 300     # training rows used as the reference for drift
DRIFT_QUANTILE = 0.95    # a row is "unlike" if farther than this from the reference
WINDOW = 10              # recent rows examined for drift
MIN_UNLIKE = 6           # unlike rows in the window needed to warn


def _objectives(row: dict, gmax: list[str], gmin: list[str]) -> list[float]:
    """Signed goal vector: larger is better on every axis. A missing or
    non-numeric goal counts as the worst possible value."""
    vec = []
    for c in gmax:
        v = _num(row.get(c))
        vec.append(v if v is not None else -math.inf)
    for c in gmin:
        v = _num(row.get(c))
        vec.append(-v if v is not None else -math.inf)
    return vec


def _dominates(a: list[float], b: list[float]) -> bool:
    """True if a is at least as good as b on every goal and better on one."""
    strictly = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strictly = True
    return strictly


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) a row if no training row, and no earlier row of the stream,
    dominates it. That makes it a member of the Pareto front of everything seen
    so far, which is the online form of step1.best. Each decision uses only
    train_csv and the rows before it."""
    header, train = _read(str(train_csv))
    _, gmax, gmin = _columns(header)
    if not gmax and not gmin:
        return [False for _ in rows]

    pool = [_objectives(r, gmax, gmin) for r in train]
    decisions: list[bool] = []
    for row in rows:
        vec = _objectives(row, gmax, gmin)
        keep = not any(_dominates(p, vec) for p in pool)
        decisions.append(keep)
        pool.append(vec)
    return decisions


def _fit(train: list[dict], inputs: list[str]) -> list[tuple]:
    """Learn how to encode each input column from the training rows only."""
    spec = []
    for c in inputs:
        vals = [r.get(c) for r in train]
        if all(isinstance(v, float) for v in vals):
            fin = [v for v in vals if math.isfinite(v)]
            lo = min(fin) if fin else 0.0
            hi = max(fin) if fin else 0.0
            spec.append(("num", lo, hi - lo))
        else:
            cats = {s: i for i, s in enumerate(sorted({str(v) for v in vals}))}
            spec.append(("cat", cats, max(1, len(cats) - 1)))
    return spec


def _encode(row: dict, inputs: list[str], spec: list[tuple]) -> list[float]:
    """Map a row's input cells to [0, 1] using the training-fitted spec.
    Unseen categories map to 1.0."""
    out = []
    for c, s in zip(inputs, spec):
        v = row.get(c)
        if s[0] == "num":
            _, lo, span = s
            x = float(v) if isinstance(v, float) and math.isfinite(v) else lo
            out.append(min(1.0, max(0.0, (x - lo) / span)) if span > 0 else 0.0)
        else:
            _, cats, top = s
            idx = cats.get(str(v))
            out.append(1.0 if idx is None else idx / top)
    return out


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the recent rows have stopped
    looking like the training rows, or None if they never do.

    The reference is a seeded sample of training input rows, encoded to [0, 1].
    A training row is "unlike" the rest of the reference if its nearest
    neighbour is farther than the DRIFT_QUANTILE of such leave-one-out
    distances. A new row is unlike if its nearest reference row is farther
    than that same limit. Warn at the first row where at least MIN_UNLIKE of
    the last WINDOW rows are unlike. Requiring several unlike rows in a window
    keeps a single odd flag set from triggering a warning.
    """
    header, train = _read(str(train_csv))
    inputs, _, _ = _columns(header)
    if len(train) < 2 or not inputs:
        return None

    spec = _fit(train, inputs)
    if len(train) <= REFERENCE_ROWS:
        sample = train
    else:
        sample = random.Random(_SEED).sample(train, REFERENCE_ROWS)
    ref = [_encode(r, inputs, spec) for r in sample]

    loo = [
        min(_dist2(x, y) for j, y in enumerate(ref) if j != i)
        for i, x in enumerate(ref)
    ]
    limit = _quantile(loo, DRIFT_QUANTILE)

    recent: deque[bool] = deque(maxlen=WINDOW)
    for i, row in enumerate(rows):
        x = _encode(_fill(row, train, inputs), inputs, spec)
        unlike = min(_dist2(x, y) for y in ref) > limit
        recent.append(unlike)
        if len(recent) == WINDOW and sum(recent) >= MIN_UNLIKE:
            return i
    return None

