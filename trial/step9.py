"""Step 9: spot. Suspicion scores for benchmark rows.

Standard library only. The cached training model and kNN predictor are reused
from step2, the input encoding, input-space nearest-neighbour distances and
reference distances from step4.

odd() adds up four kinds of evidence for each row. Higher means more suspicious:
  - input_part: the flag set is far from every training flag set. The distance
    is divided by the typical leave-one-out nearest-neighbour distance of the
    training data, so a typical row scores about 1.
  - goal_part: a goal value disagrees with what the nearest training builds
    predict, in units of the calibrated residual spread from step2. Only the
    worst goal counts.
  - range_part: a goal value lies outside the training range, in units of that
    range (or of the residual spread when the range is zero).
  - unseen_part and bad_cell: a flag-like input (at most FLAG_LIKE distinct
    training values) takes a value never seen in training or is missing, and a
    goal cell is missing, not a number, or not finite.
"""

import math
import statistics

from step2 import _model, _predict
from step4 import _encode, _nearest_sq, _reference


FLAG_LIKE = 10          # inputs with at most this many training values count as flags
UNSEEN_COST = 1.0       # points per flag-like input that is unseen or missing
BAD_CELL = 10.0         # points per goal cell that is missing, not a number, or not finite
SIGMA_PER_BAND = 3.29   # width of a 5%..95% band of a normal, in standard deviations


def _input_scale(ref):
    """Typical nearest-neighbour distance of training rows: the median, or the
    smallest positive distance when the median is zero (duplicate flag sets)."""
    if not ref:
        return None
    med = statistics.median(ref)
    if med > 0:
        return med
    positive = [d for d in ref if d > 0]
    return min(positive) if positive else 1.0


def _goal_sigmas(model):
    """Per goal, a residual spread: the calibrated 5..95% band divided by
    SIGMA_PER_BAND when usable, else the training standard deviation, else 1."""
    sigmas = []
    for g, (lo, hi) in enumerate(model["bands"]):
        s = 0.0
        if math.isfinite(lo) and math.isfinite(hi):
            s = (hi - lo) / SIGMA_PER_BAND
        if not s > 0:
            vals = [t[g] for t in model["targets"]]
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        sigmas.append(s if s > 0 else 1.0)
    return sigmas


def _flag_values(model):
    """For each flag-like input column, the set of values seen in training."""
    flags = {}
    for c in model["inputs"]:
        vals = {r[c] for r in model["rows"]}
        if len(vals) <= FLAG_LIKE:
            flags[c] = vals
    return flags


def _goal_cells(row, goals):
    """Per goal, the raw float value, or None when missing, not a number or not
    finite. Also returns how many cells were bad."""
    cells = []
    bad = 0
    for name, _ in goals:
        try:
            x = float(row.get(name))
        except (TypeError, ValueError):
            x = None
        if x is None or not math.isfinite(x):
            cells.append(None)
            bad += 1
        else:
            cells.append(x)
    return cells, bad


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    model = _model(train_csv)
    goals = model["goals"]
    targets = model["targets"]
    if not rows:
        return []
    if not targets:
        return [math.inf] * len(rows)

    m = len(goals)
    points = model["points"]
    ref = _reference(train_csv, model)
    scale = _input_scale(ref)
    sigmas = _goal_sigmas(model)
    flags = _flag_values(model)
    lo = [min(t[g] for t in targets) for g in range(m)]
    hi = [max(t[g] for t in targets) for g in range(m)]

    out = []
    for row in rows:
        score = 0.0
        q = _encode(model, row)

        # Input-space novelty, relative to the typical training distance.
        if scale is not None and points:
            score += math.sqrt(_nearest_sq(q, points)) / scale

        # Flag-like inputs with values never seen in training, or missing.
        for c, vals in flags.items():
            v = row.get(c)
            if v is None or v not in vals:
                score += UNSEEN_COST

        # Goal cells: bad cells, range excess, and disagreement with neighbours.
        cells, bad = _goal_cells(row, goals)
        score += BAD_CELL * bad

        if m and any(x is not None for x in cells):
            pred = _predict(q, points, targets)
            resid = []
            for g, x in enumerate(cells):
                if x is None:
                    continue
                span = hi[g] - lo[g]
                unit = span if span > 0 else sigmas[g]
                score += max(lo[g] - x, x - hi[g], 0.0) / unit
                if pred is not None:
                    resid.append(abs(x - pred[g]) / sigmas[g])
            if resid:
                score += max(resid)

        out.append(float(score))
    return out

