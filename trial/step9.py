"""step9.py: flag suspicious benchmark results and unusual flag sets.

odd(train_csv, rows) gives every row a suspicion score (higher means more
suspicious). It has two parts, both measured against train_csv:

  1. Unusual flags: the distance from the row's inputs to the nearest training
     flag set, in units of the typical leave-one-out nearest distance inside
     the training data (a sample of it). Flag sets far from everything
     measured score high.
  2. Glitchy goals: each goal is compared with the median goal of the nearest
     training flag sets (step2.guess), in units of that goal's spread across
     the training data. A value far from what similar builds achieve scores
     high. A missing, non-numeric or non-finite goal scores MISSING_SCORE.

The score is the sum of the two parts. A typical training-like row scores
about 1. Standard library only.
"""
import math
import random
import statistics

from step1 import _load, _columns
from step2 import guess

SAMPLE = 200          # training rows sampled to set the typical nearest distance
MISSING_SCORE = 1e6   # goal part of a row whose goal cell is missing or unusable
SEED = 1


def _vec(row, inputs):
    """Input values of a row as floats, None where missing or non-numeric."""
    return [float(row[c]) if isinstance(row.get(c), float) else None
            for c in inputs]


def _input_dist(x, r, spans):
    """Root-mean-square difference between two flag vectors, each input scaled
    by its training span. A missing input counts as a full mismatch."""
    s = 0.0
    for a, b, span in zip(x, r, spans):
        if a is None or b is None:
            d = 1.0
        elif span > 0:
            d = (a - b) / span
        else:
            d = 0.0 if a == b else 1.0
        s += d * d
    return math.sqrt(s / len(spans))


def _spread(vals):
    """Interquartile range of a list of floats, or 0.0 if it cannot be formed."""
    if len(vals) < 2:
        return 0.0
    q = statistics.quantiles(vals, n=4)
    return q[2] - q[0]


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row of `rows` (goals included): higher means more
    likely a glitch, a typo, or a flag set unlike anything in train_csv."""
    rows = list(rows)
    if not rows:
        return []
    cols, train = _load(train_csv)
    inputs, maxi, mini = _columns(cols, train)
    goals = maxi + mini
    if not train:
        # Nothing to compare against.
        return [0.0] * len(rows)

    # Part 1: unusual flag sets.
    spans = []
    X = []
    ref = 1.0
    if inputs:
        spans = [max(r[c] for r in train) - min(r[c] for r in train)
                 for c in inputs]
        X = [_vec(r, inputs) for r in train]
        # Typical nearest distance inside the training data, from a sample.
        # Zero distances (duplicate flag sets) are left out, so the reference
        # is not collapsed to zero.
        rng = random.Random(SEED)
        idx = list(range(len(train)))
        if len(idx) > SAMPLE:
            idx = rng.sample(idx, SAMPLE)
        ref_d = []
        for i in idx:
            d = min((_input_dist(X[i], X[j], spans)
                     for j in range(len(X)) if j != i), default=0.0)
            if d > 0:
                ref_d.append(d)
        ref = statistics.median(ref_d) if ref_d else 1.0

    def input_part(row):
        if not inputs:
            return 0.0
        x = _vec(row, inputs)
        nearest = min(_input_dist(x, t, spans) for t in X)
        return nearest / ref

    # Part 2: glitchy goals. Each goal is scaled by its spread in the training data.
    scale = {}
    for g in goals:
        vals = [r[g] for r in train
                if isinstance(r[g], float) and math.isfinite(r[g])]
        s = _spread(vals)
        if s <= 0 and len(vals) > 1:
            s = statistics.pstdev(vals)
        if s <= 0 and vals:
            s = abs(statistics.median(vals)) * 0.01
        scale[g] = s if s > 0 else 1.0

    def goal_part(row):
        if not goals:
            return 0.0
        for g in goals:
            v = row.get(g)
            if not isinstance(v, float) or not math.isfinite(v):
                return MISSING_SCORE
        flags = {c: row.get(c) for c in inputs}
        pred = guess(train_csv, flags)
        worst = 0.0
        for g in goals:
            est, lo, hi = pred[g]
            if math.isinf(lo) or math.isinf(hi):
                continue  # no similar training builds to compare with
            v = row[g]
            worst = max(worst, abs(v - est) / scale[g])
        return worst

    out = []
    for row in rows:
        out.append(float(input_part(row) + goal_part(row)))
    return out

