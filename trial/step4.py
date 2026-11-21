"""Step 4: streaming flow control for nightly benchmark results.

stream() keeps each incoming row that is not Pareto-dominated (on the goals)
by the training rows or by the rows that arrived before it.

drift() compares each incoming row's input-space neighbourhood distance with
the training set. It warns when most rows in a sliding window are far from
the training data, which is the same notion of "far" that step2 uses.
"""

import heapq
import math
from collections import deque

from step1 import _load, _split_columns

K = 15
FAR_THRESHOLD = 0.25
WINDOW = 20
WINDOW_FRAC = 0.5
REF_SAMPLE = 150


def _as_float(value):
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dominates(a, b):
    """True if a Pareto-dominates b (all dimensions minimised)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _goal_vector(row, maximise, minimise):
    """Minimisation vector for a row, or None if any goal is missing or non-finite."""
    vec = []
    for c in maximise:
        v = _as_float(row.get(c))
        if v is None or not math.isfinite(v):
            return None
        vec.append(-v)
    for c in minimise:
        v = _as_float(row.get(c))
        if v is None or not math.isfinite(v):
            return None
        vec.append(v)
    return vec


def stream(train_csv, rows):
    """Return one keep flag per incoming row.

    A row is kept (True) when no training row and no earlier incoming row
    Pareto-dominates it on the goals ('+' maximised, '-' minimised). Goals
    are normalised with the training ranges. Each row is judged using only
    train_csv and the rows before it.
    """
    columns, train = _load(train_csv)
    maximise, minimise, _ = _split_columns(columns)
    if not (maximise or minimise):
        return [False] * len(rows)

    ref = []
    for r in train:
        g = _goal_vector(r, maximise, minimise)
        if g is not None:
            ref.append(g)
    if not ref:
        return [False] * len(rows)

    m = len(ref[0])
    lo = [min(g[j] for g in ref) for j in range(m)]
    hi = [max(g[j] for g in ref) for j in range(m)]

    def norm(g):
        return [
            (g[j] - lo[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else 0.0
            for j in range(m)
        ]

    seen = [norm(g) for g in ref]
    out = []
    for row in rows:
        g = _goal_vector(row, maximise, minimise)
        if g is None:
            out.append(False)
            continue
        x = norm(g)
        keep = not any(_dominates(s, x) for s in seen)
        out.append(keep)
        seen.append(x)
    return out


def _spec(rows, inputs):
    spec = {}
    for c in inputs:
        vals = [r.get(c) for r in rows]
        if vals and all(isinstance(v, float) for v in vals):
            spec[c] = ("num", min(vals), max(vals))
        else:
            spec[c] = ("cat", None, None)
    return spec


def _distance(a, b, inputs, spec):
    """Sum of per-column differences; numeric columns are scaled to [0, 1]."""
    d = 0.0
    for c in inputs:
        kind, lo, hi = spec[c]
        x = a.get(c)
        y = b.get(c)
        if kind == "num":
            fx = _as_float(x)
            fy = _as_float(y)
            if fx is None or fy is None:
                d += 1.0
            else:
                span = hi - lo
                d += abs(fx - fy) / span if span > 0 else 0.0
        else:
            d += 0.0 if str(x) == str(y) else 1.0
    return d


def drift(train_csv, rows):
    """Return the index into rows of the first drift warning, or None.

    Each row gets a score: the mean normalised distance to its K nearest
    training rows. A row is "far" when that score exceeds FAR_THRESHOLD.
    Training rows are scored leave-one-out to estimate the baseline far rate.
    A warning is raised at the first index where the far rate over the last
    WINDOW rows exceeds max(WINDOW_FRAC, baseline + 0.25). Only rows up to
    the current index are used.
    """
    columns, train = _load(train_csv)
    _, _, inputs = _split_columns(columns)
    n = len(train)
    if n == 0 or not rows:
        return None

    spec = _spec(train, inputs)
    width = max(1, len(inputs))

    def score(row, exclude, k):
        dists = [
            _distance(t, row, inputs, spec)
            for j, t in enumerate(train)
            if j != exclude
        ]
        if not dists or k <= 0:
            return 0.0
        near = heapq.nsmallest(min(k, len(dists)), dists)
        return sum(near) / (len(near) * width)

    stride = max(1, n // REF_SAMPLE)
    ref = [score(train[i], i, min(K, n - 1)) for i in range(0, n, stride)]
    base = sum(1 for s in ref if s > FAR_THRESHOLD) / len(ref)
    thresh = max(WINDOW_FRAC, base + 0.25)

    window = deque(maxlen=WINDOW)
    for i, row in enumerate(rows):
        window.append(score(row, None, min(K, n)) > FAR_THRESHOLD)
        if len(window) == WINDOW and sum(window) / WINDOW > thresh:
            return i
    return None

