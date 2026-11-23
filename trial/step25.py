"""Step 25: client-facing blame for two customer questions.

blame(train_csv): input column names, most to blame for bad rows first.

This version answers the client's two questions directly:
  * "Why do some customers never respond to campaigns?" uses RESPONSE+.
    A customer is bad here when RESPONSE+ is not 1 (0 or missing).
  * "What do the customers who complain have in common?" uses COMPLAIN-.
    A customer is bad here when COMPLAIN- is above 0 (or missing).

Each input gets a signed score per goal: for binary inputs, the difference in
how often the input is on among bad rows versus good rows; for other inputs,
the Spearman-style correlation between the input's ranks and the bad indicator.
The two goal scores are summed per input. Higher total means more blame, and
ties keep the original column order. If neither goal column is present, this
falls back to step5.blame (all goals together).
"""

import csv
import math

import numpy as np

import step5

RESPONSE = "RESPONSE+"
COMPLAIN = "COMPLAIN-"


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _split_columns(cols):
    goal_max = [c for c in cols if c.endswith("+")]
    goal_min = [c for c in cols if c.endswith("-")]
    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]
    return inputs, goal_max, goal_min


def _average_ranks(values):
    """Ranks 1..N with ties replaced by their average rank."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[np.asarray(inverse).reshape(-1)]


def _bad_mask(goal, rows):
    """True for the customers that are bad on this goal."""
    vals = np.array([_to_float(r.get(goal)) for r in rows], dtype=float)
    finite = np.isfinite(vals)
    if goal == RESPONSE:
        # Bad = never responds: anything other than a positive response.
        return ~(finite & (vals > 0))
    # COMPLAIN-: bad = complained, or missing (counts as worst).
    return ~finite | (vals > 0)


def _score(vals, bad):
    """Signed score: positive means the input goes with the bad group.
    Returns None if one of the two groups is empty."""
    good = ~bad
    if not bad.any() or not good.any():
        return None
    uniq = np.unique(vals)
    if len(uniq) <= 1:
        return 0.0
    if len(uniq) == 2:
        on = vals == uniq[-1]
        return float(on[bad].mean() - on[good].mean())
    rv = _average_ranks(vals)
    rb = bad.astype(float)
    if rv.std() == 0 or rb.std() == 0:
        return 0.0
    s = float(np.corrcoef(rv, rb)[0, 1])
    return s if math.isfinite(s) else 0.0


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows first.

    Bad rows are defined by the client's two questions: non-responders
    (RESPONSE+ not 1) and complainers (COMPLAIN- above 0). Inputs are scored
    per goal and the scores are summed. Ties keep the original column order.
    """
    cols, rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    n = len(rows)

    if not inputs:
        return []
    if n == 0:
        return list(inputs)

    goals = [g for g in (RESPONSE, COMPLAIN) if g in cols]
    if not goals:
        return [str(c) for c in step5.blame(train_csv)]

    # Input values, missing treated as 0.0 (as in step5).
    values = {}
    for c in inputs:
        v = np.array([_to_float(r.get(c)) for r in rows], dtype=float)
        values[c] = np.where(np.isnan(v), 0.0, v)

    totals = np.zeros(len(inputs), dtype=float)
    used = 0
    for g in goals:
        bad = _bad_mask(g, rows)
        if not bad.any() or bad.all():
            continue
        used += 1
        for idx, c in enumerate(inputs):
            s = _score(values[c], bad)
            if s is not None:
                totals[idx] += s

    if used == 0:
        return [str(c) for c in step5.blame(train_csv)]

    order = sorted(range(len(inputs)), key=lambda i: (-totals[i], i))
    return [inputs[i] for i in order]

