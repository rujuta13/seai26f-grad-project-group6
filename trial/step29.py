"""step29.py: flag customer records that look like typos or glitches, or that are unlike
anyone in the training data.

odd() is step19.odd (goal-glitch scoring, unseen categories and unparseable values scored
inf, novelty measured against the training customers with identifier columns left out),
plus one new check. Novelty is a distance across all traits, so a single impossible value
(an age of 250, an income of 10 million) can be hidden by the other traits. The new check
scores each numeric trait on its own, as a robust z-score against the training customers,
and the row keeps the larger of the two scores.

Helpers are reused from step1 (loading, classification, value coercion), step9 (robust
scaling), step11 (identifier filter) and step19 (the full odd() score).
"""
from __future__ import annotations

import math

import numpy as np

from step1 import _classify, _load, _to_value
from step9 import _robust_scale
from step11 import _trait_inputs
from step19 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more suspicious.

    A score of inf marks a row with a missing or non-finite goal, an unparseable or
    non-finite numeric input, or a categorical input value never seen in train_csv.
    Otherwise the score is the larger of the step19 score (goal glitch or unlike the
    training data as a whole) and the largest per-trait robust z-score of the numeric
    inputs (one impossible value on its own).
    """
    if not rows:
        return []

    base = _odd(train_csv, rows)

    header, train_rows = _load(train_csv)
    _, _, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)

    # Per-trait robust location and spread, for numeric trait columns only.
    num_cols = []
    for c in inputs:
        vals = [_to_value(r[c]) for r in train_rows]
        if vals and all(isinstance(v, float) and math.isfinite(v) for v in vals):
            num_cols.append((c, _robust_scale(np.array(vals, dtype=float))))

    scores: list[float] = []
    for row, b in zip(rows, base):
        if not math.isfinite(b):
            scores.append(b)
            continue
        worst = 0.0
        for c, (med, scale) in num_cols:
            raw = row.get(header[c])
            if raw is None:
                continue
            v = _to_value(raw)
            if isinstance(v, float) and math.isfinite(v):
                worst = max(worst, abs(v - med) / scale)
        scores.append(float(max(b, worst)))

    return scores

