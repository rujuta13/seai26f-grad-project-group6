"""step9.py: spot glitches and odd flag sets among benchmark results.

Standard library only (Python 3.13). Reuses step2 (kNN goal estimates via
guess() and cell parsing via _num) and step4 (training context, cell access
via _raw, and each row's distance to the training inputs via _row_distance).

odd() returns one suspicion score per row, goals included. Higher means more
likely a glitch, a typo, or unlike anything in the training file. Each score
adds:
  - input novelty: distance from the row's inputs to the nearest training
    row, relative to the typical leave-one-out distance in training;
  - unseen input cells: blank, text or infinite cells whose value never
    occurs in training;
  - goal anomalies: blank, text or infinite goal cells; goal values outside
    the training range; and goal values far from the kNN estimate of
    step2.guess, given the row's inputs.
Each row is scored on its own against the training file, so rows in the
same batch do not influence each other's scores.
"""
from __future__ import annotations

import math
import os
import statistics

from step2 import _num, guess
from step4 import _context, _raw, _row_distance

_UNSEEN_PENALTY = 1.0    # per input cell with a value never seen in training
_MISSING_PENALTY = 3.0   # per goal cell that is blank, text or infinite
_MAX_TERM = 20.0         # cap on any single damped term
_MIN_SPAN_SHARE = 0.05   # goal tolerance is at least this share of the training range


def _damp(x: float) -> float:
    """log(1 + x), capped at _MAX_TERM. NaN counts as the cap."""
    if math.isnan(x):
        return _MAX_TERM
    if x <= 0.0:
        return 0.0
    return min(math.log1p(x), _MAX_TERM)


def _finite(raw) -> float | None:
    """Numeric value of a cell, or None if blank, text, NaN or infinite."""
    v = _num(raw)
    if v is None or not math.isfinite(v):
        return None
    return v


def _scale_in(ctx: dict) -> float:
    """Typical leave-one-out training distance, used to scale input novelty."""
    ref = [d for d in ctx["ref"] if math.isfinite(d)]
    pos = [d for d in ref if d > 0.0]
    if pos:
        return statistics.median(pos)
    return 1.0


def _unseen_inputs(row, ctx: dict) -> int:
    """Input cells that are blank, text or infinite and never seen in training."""
    tables = ctx["tables"]
    count = 0
    for name, j in zip(ctx["input_names"], ctx["inputs"]):
        raw = _raw(row, name, j)
        if _finite(raw) is not None:
            continue
        text = "" if raw is None else str(raw)
        table = tables.get(j, {})
        if text not in table and text.strip() not in table:
            count += 1
    return count


def _goal_term(row, ctx: dict,
               est: dict[str, tuple[float, float, float]]) -> float:
    """Sum of goal anomaly terms for one row."""
    total = 0.0
    for k, (name, j) in enumerate(zip(ctx["goal_names"], ctx["goal_idx"])):
        v = _finite(_raw(row, name, j))
        if v is None:
            total += _MISSING_PENALTY
            continue

        glo = ctx["glo"][k]
        ghi = ctx["ghi"][k]
        span = 0.0
        if glo is not None and ghi is not None:
            span = ghi - glo
            if not math.isfinite(span) or span < 0.0:
                span = 0.0

        excess = 0.0
        if span > 0.0 and glo is not None and ghi is not None:
            below = max(0.0, glo - v)
            above = max(0.0, v - ghi)
            excess = (below + above) / span

        e, lo, hi = est.get(name, (math.nan, math.nan, math.nan))
        if math.isnan(e):
            dev = 0.0
        else:
            base = max(hi - lo, _MIN_SPAN_SHARE * span,
                       1e-9 * max(1.0, abs(e)))
            dev = abs(v - e) / base

        total += _damp(dev) + _damp(excess)
    return total


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in rows (goals included).

    Higher means more likely a glitch, a typo, or unlike anything in
    train_csv. Scores are finite and non-negative. If train_csv has no
    usable rows, every row gets the maximum score.
    """
    key = os.fspath(train_csv)
    ctx = _context(key)
    rows = list(rows)
    if not rows:
        return []
    if not ctx["Z"]:
        return [_MAX_TERM] * len(rows)

    scale = _scale_in(ctx)
    scores: list[float] = []
    for row in rows:
        inputs_map = {
            name: _raw(row, name, j)
            for name, j in zip(ctx["input_names"], ctx["inputs"])
        }
        input_term = _damp(_row_distance(row, ctx) / scale)
        unseen = _unseen_inputs(row, ctx)
        est = guess(key, inputs_map)
        score = (input_term
                 + _UNSEEN_PENALTY * unseen
                 + _goal_term(row, ctx, est))
        scores.append(float(score))
    return scores

