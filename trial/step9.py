import csv
import math

import numpy as np


_BIG = 1e6  # z-score assigned to a missing or non-numeric goal value


def _to_value(cell):
    if cell is None:
        return None
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, (int, float)):
        return float(cell)
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [{k: _to_value(v) for k, v in r.items()} for r in reader]
    return header, rows


def _kind(name):
    if not name:
        return "ignore"
    last = name[-1]
    if last == "X":
        return "ignore"
    if last == "+":
        return "max"
    if last == "-":
        return "min"
    return "input"


def _as_finite(v):
    """Return v as a finite float, or None if missing or not numeric."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def _input_specs(header, train):
    """Per input column: (name, labels or None, fill value)."""
    specs = []
    for c in header:
        if _kind(c) != "input":
            continue
        values = [r.get(c) for r in train]
        present = [v for v in values if not _is_missing(v)]
        numeric = bool(present) and all(_as_finite(v) is not None for v in present)
        if numeric:
            nums = [_as_finite(v) for v in present]
            fill = float(np.mean(nums))
            specs.append((c, None, fill))
        else:
            labels = {}
            for v in values:
                key = "" if _is_missing(v) else str(v)
                if key not in labels:
                    labels[key] = float(len(labels))
            specs.append((c, labels, 0.0))
    return specs


def _encode(row, specs):
    out = np.empty(len(specs), dtype=float)
    for j, (c, labels, fill) in enumerate(specs):
        v = row.get(c)
        if labels is None:
            f = _as_finite(v)
            out[j] = fill if f is None else f
        else:
            key = "" if _is_missing(v) else str(v)
            out[j] = labels.get(key, float(len(labels)))
    return out


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows`; higher means more likely a glitch,
    a typo, or unlike anything in train_csv.

    Two signals are combined on a log scale:
      * goal outlier: the largest robust z-score (median/MAD, fitted on the
        training builds) over all goal columns. A missing or non-numeric goal
        gets a very large score.
      * input novelty: nearest-neighbour distance to the training inputs in
        standardised input space, relative to the typical leave-one-out
        nearest-neighbour distance within the training set.
    With no training rows there is nothing to compare against, so all scores
    are 0.0.
    """
    rows = list(rows)
    if not rows:
        return []

    header, train = _load(train_csv)
    n = len(train)
    if n == 0:
        return [0.0] * len(rows)

    # ---- input novelty
    in_scores = np.zeros(len(rows), dtype=float)
    specs = _input_specs(header, train)
    if specs and n >= 2:
        X = np.vstack([_encode(r, specs) for r in train])
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (X - mu) / sd

        diff = Z[:, None, :] - Z[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(D, np.inf)
        base = D.min(axis=1)

        scale = float(np.median(base))
        if not scale > 0:
            positive = base[base > 0]
            scale = float(positive.mean()) if positive.size else 1.0

        Zr = np.vstack([(_encode(r, specs) - mu) / sd for r in rows])
        Dr = np.sqrt(((Zr[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1))
        in_scores = Dr.min(axis=1) / scale

    # ---- goal outliers
    goals = [c for c in header if _kind(c) in ("max", "min")]
    goal_scores = np.zeros(len(rows), dtype=float)
    if goals:
        per_goal = []
        for g in goals:
            ys = np.array(
                [v for v in (_as_finite(r.get(g)) for r in train) if v is not None],
                dtype=float,
            )
            if ys.size == 0:
                continue
            med = float(np.median(ys))
            mad = 1.4826 * float(np.median(np.abs(ys - med)))
            if not mad > 0:
                mad = float(ys.std())
            if not mad > 0:
                mad = 1.0

            z = np.empty(len(rows), dtype=float)
            for i, r in enumerate(rows):
                v = _as_finite(r.get(g))
                if v is None:
                    z[i] = _BIG
                else:
                    z[i] = min(abs(v - med) / mad, _BIG)
            per_goal.append(z)
        if per_goal:
            goal_scores = np.max(np.vstack(per_goal), axis=0)

    score = np.log1p(goal_scores) + np.log1p(in_scores)
    score = np.nan_to_num(score, nan=_BIG, posinf=_BIG, neginf=0.0)
    return [float(s) for s in score]

