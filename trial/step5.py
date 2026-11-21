import csv
import math

import numpy as np


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
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[inverse]


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    A row's badness is its average normalised rank across all goals, with
    every goal minimised and missing values counted as worst. Rows in the top
    quartile of badness are "bad", rows in the bottom quartile are "good".

    Binary inputs are scored by how much more often they are on among bad rows
    than among good rows. Other inputs are scored by the Spearman correlation
    between their rank and the badness rank. Higher score means more blame.
    Ties keep the original column order.
    """
    cols, rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max
    n = len(rows)

    if not inputs:
        return []
    if n == 0 or not goals:
        return list(inputs)

    # Minimisation matrix: every goal minimised, missing = worst (+inf).
    G = np.empty((n, len(goals)), dtype=float)
    for j, c in enumerate(goals):
        vals = np.array([_to_float(r.get(c)) for r in rows], dtype=float)
        vals = np.where(np.isnan(vals), np.inf, vals)
        if c.endswith("+"):
            vals = -vals
        G[:, j] = vals

    # Badness: mean normalised rank across goals (0 = best, 1 = worst).
    badness = np.zeros(n, dtype=float)
    for j in range(G.shape[1]):
        r = _average_ranks(G[:, j])
        badness += (r - 1.0) / max(n - 1, 1)
    badness /= G.shape[1]

    if np.all(badness == badness[0]):
        return list(inputs)

    q75 = float(np.quantile(badness, 0.75))
    q25 = float(np.quantile(badness, 0.25))
    bad = badness >= q75
    good = (~bad) & (badness <= q25)
    if not good.any():
        good = ~bad
    if not bad.any() or not good.any():
        return list(inputs)

    rank_bad = _average_ranks(badness)

    scored = []
    for idx, c in enumerate(inputs):
        vals = np.array([_to_float(r.get(c)) for r in rows], dtype=float)
        vals = np.where(np.isnan(vals), 0.0, vals)
        uniq = np.unique(vals)

        if len(uniq) <= 1:
            score = 0.0
        elif len(uniq) == 2:
            on = vals == uniq[-1]
            score = float(on[bad].mean() - on[good].mean())
        else:
            rv = _average_ranks(vals)
            if rv.std() == 0 or rank_bad.std() == 0:
                score = 0.0
            else:
                score = float(np.corrcoef(rv, rank_bad)[0, 1])
                if not math.isfinite(score):
                    score = 0.0

        scored.append((-score, idx, c))

    scored.sort()
    return [c for _, _, c in scored]

