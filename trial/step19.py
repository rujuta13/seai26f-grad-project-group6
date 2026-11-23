import csv
import math

import numpy as np


_BIG = 1e6           # z-score assigned to a missing or non-numeric goal value
_ID_FRACTION = 0.9   # a column this unique is treated as an identifier
_MAX_LEVELS = 50     # categorical columns with more levels are skipped


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


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


def _num(cell):
    """Return cell as a finite float, or None if missing or not numeric."""
    if cell is None or isinstance(cell, bool):
        return None
    try:
        f = float(cell)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def _key(v):
    return "" if _missing(v) else str(v)


def _input_specs(header, train):
    """Per usable input column: (name, kind, meta).

    kind "num": meta = (mean, sd) for standardisation.
    kind "cat": meta = {level: index} for one-hot encoding.
    Identifier-like columns (names, urls, row numbers) are skipped so that
    a new name or url is not mistaken for an unusual build.
    """
    n = len(train)
    specs = []
    for c in header:
        if _kind(c) != "input":
            continue
        raw = [r.get(c) for r in train]
        present = [v for v in raw if not _missing(v)]
        nums = [_num(v) for v in present]
        numeric = bool(present) and all(x is not None for x in nums)
        distinct = len({_key(v) for v in raw})
        if n > 0 and distinct > _ID_FRACTION * n and (
            not numeric or all(float(x).is_integer() for x in nums)
        ):
            continue
        if numeric:
            arr = np.asarray(nums, dtype=float)
            mean = float(arr.mean())
            sd = float(arr.std())
            if not sd > 0:
                sd = 1.0
            specs.append((c, "num", (mean, sd)))
        else:
            levels = sorted({_key(v) for v in raw})
            if len(levels) > _MAX_LEVELS:
                continue
            specs.append((c, "cat", {k: i for i, k in enumerate(levels)}))
    return specs


def _encode(rowlist, specs):
    m = len(rowlist)
    blocks = []
    for c, kind, meta in specs:
        if kind == "num":
            mean, sd = meta
            col = np.zeros(m, dtype=float)
            for i, r in enumerate(rowlist):
                x = _num(r.get(c))
                col[i] = 0.0 if x is None else (x - mean) / sd
            blocks.append(col[:, None])
        else:
            M = np.zeros((m, len(meta)), dtype=float)
            for i, r in enumerate(rowlist):
                j = meta.get(_key(r.get(c)))
                if j is not None:
                    M[i, j] = 1.0
            blocks.append(M)
    if not blocks:
        return np.zeros((m, 0), dtype=float)
    return np.hstack(blocks)


def _sqdist(A, B):
    a2 = (A * A).sum(axis=1)[:, None]
    b2 = (B * B).sum(axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def _novelty(header, train, rows):
    """Nearest-neighbour distance to training inputs, relative to the typical
    leave-one-out nearest-neighbour distance within the training set."""
    out = np.zeros(len(rows), dtype=float)
    specs = _input_specs(header, train)
    if not specs or len(train) < 2:
        return out
    Z = _encode(train, specs)
    if Z.shape[1] == 0:
        return out

    D = _sqdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    base = np.sqrt(D.min(axis=1))

    scale = float(np.median(base))
    if not scale > 0:
        positive = base[base > 0]
        scale = float(positive.mean()) if positive.size else 1.0

    Zr = _encode(rows, specs)
    dist = np.sqrt(_sqdist(Zr, Z).min(axis=1))
    return dist / scale


def _goal_outlier(header, train, rows):
    """Largest robust z-score (median/MAD on training goals) over all goals.
    A missing or non-numeric goal gets a very large score."""
    goals = [c for c in header if _kind(c) in ("max", "min")]
    out = np.zeros(len(rows), dtype=float)
    if not goals:
        return out
    for g in goals:
        ys = [x for x in (_num(r.get(g)) for r in train) if x is not None]
        if not ys:
            continue
        ys = np.asarray(ys, dtype=float)
        med = float(np.median(ys))
        mad = 1.4826 * float(np.median(np.abs(ys - med)))
        if not mad > 0:
            mad = float(ys.std())
        if not mad > 0:
            mad = 1.0
        z = np.empty(len(rows), dtype=float)
        for i, r in enumerate(rows):
            v = _num(r.get(g))
            z[i] = _BIG if v is None else min(abs(v - med) / mad, _BIG)
        out = np.maximum(out, z)
    return out


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more
    likely a glitch, a typo, or unlike anything in train_csv.

    Two signals are combined on a log scale:
      * goal outlier: the largest robust z-score over the goal columns,
        fitted on the training builds. A missing or non-numeric goal scores
        very high.
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
    if not train:
        return [0.0] * len(rows)

    in_score = _novelty(header, train, rows)
    goal_score = _goal_outlier(header, train, rows)

    score = np.log1p(goal_score) + np.log1p(in_score)
    score = np.where(np.isfinite(score), score, _BIG)
    return [float(s) for s in score]

