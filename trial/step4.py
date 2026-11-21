"""Step 4: streaming keep/drift decisions for nightly benchmark results.

Standard library only. Reuses the Pareto dominance and encoding helpers from
step1, the cached training model from step2, and the permutation test from
step3.

stream: a new row is kept when no training row and no earlier row of the
stream is at least as good on every goal, with one strictly better (or
identical). Goals are normalised with the bounds of train_csv, so the verdict
depends only on the training data and the rows already seen.

drift: each incoming row is summarised by its Euclidean distance to the
nearest training row in input space. Those distances are compared, in
non-overlapping blocks of DRIFT_BLOCK rows, with the leave-one-out nearest
neighbour distances of the training data, using step3.same. The index
returned is the last row of the first block that differs.
"""

import math

from step1 import _dominates, _input_matrix, _sqdist
from step2 import _model
from step3 import same


DRIFT_BLOCK = 30      # rows per non-overlapping drift check
CALIB_SAMPLE = 200    # training rows used for the reference distances

_ref_cache = {}


def _goal_bounds(model):
    """Per goal, the (lo, hi) of the signed training values (higher is better)."""
    bounds = []
    for g, (_, sign) in enumerate(model["goals"]):
        vals = [sign * t[g] for t in model["targets"]]
        if not vals:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((min(vals), max(vals)))
    return bounds


def _goal_vector(values, goals, bounds):
    """Goal values of one row, scaled by the training bounds (higher is better).

    A goal with zero training span keeps its raw offset from the training
    value, so a different result on that goal is still visible."""
    vec = []
    for (name, sign), (lo, hi) in zip(goals, bounds):
        span = hi - lo
        scale = span if span > 0 else 1.0
        vec.append((sign * float(values[name]) - lo) / scale)
    return vec


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) a row if no training row and no earlier stream row is at
    least as good on every goal with a strict gain on one (or an exact tie).
    Each decision uses only train_csv and the rows before it."""
    model = _model(train_csv)
    goals = model["goals"]
    if not goals:
        return [False for _ in rows]

    bounds = _goal_bounds(model)
    seen = [_goal_vector(r, goals, bounds) for r in model["rows"]]

    decisions = []
    for row in rows:
        vec = _goal_vector(row, goals, bounds)
        keep = not any(p == vec or _dominates(p, vec) for p in seen)
        decisions.append(keep)
        seen.append(vec)
    return decisions


def _nearest_sq(query, points, exclude=None):
    """Squared distance from query to its nearest point, skipping index exclude."""
    best = math.inf
    for i, p in enumerate(points):
        if i == exclude:
            continue
        d = _sqdist(query, p)
        if d < best:
            best = d
    return best


def _reference(train_csv, model):
    """Leave-one-out nearest-neighbour distances of a sample of training rows."""
    if train_csv in _ref_cache:
        return _ref_cache[train_csv]
    points = model["points"]
    n = len(points)
    step = max(1, n // CALIB_SAMPLE)
    ref = []
    for i in range(0, n, step):
        d = _nearest_sq(points[i], points, exclude=i)
        if d != math.inf:
            ref.append(math.sqrt(d))
    _ref_cache[train_csv] = ref
    return ref


def _encode(model, row):
    """Encode one row with the training codes, as step2.guess does."""
    probe = {c: row.get(c, 0.0) for c in model["inputs"]}
    return _input_matrix(model["rows"] + [probe], model["inputs"])[-1]


def drift(train_csv, rows) -> int | None:
    """Index into rows where new results first stop looking like train_csv,
    judged on input-space nearest-neighbour distances in blocks of
    DRIFT_BLOCK rows. Returns None if no block differs."""
    model = _model(train_csv)
    points = model["points"]
    if not points:
        return None
    ref = _reference(train_csv, model)
    if not ref:
        return None

    block = []
    for i, row in enumerate(rows):
        q = _encode(model, row)
        d = _nearest_sq(q, points)
        block.append(math.sqrt(d))
        if len(block) == DRIFT_BLOCK:
            if not same(block, ref):
                return i
            block = []
    return None

