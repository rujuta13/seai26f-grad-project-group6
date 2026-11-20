"""step2.py: guess the goals (performance and CPU cost) of an unbenchmarked row.

Standard library only (Python 3.13). Reuses step1's csv and MOOT helpers.

The guess is a k-nearest-neighbour estimate over the scaled input columns.
Each goal's estimate is the median of its nearest neighbours' values, and
low..high is the 10th..90th percentile of those values, so the interval
reflects how much the closest training rows disagree.
"""
from __future__ import annotations

import math
import os

from step1 import _classify, _read, _to_float

_K = 15        # neighbours used per goal
_LO_Q = 0.10   # lower bound quantile
_HI_Q = 0.90   # upper bound quantile

_cache: dict[str, dict] = {}


def _num(value) -> float | None:
    """Numeric value of a cell, or None if it is missing, NaN or text."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        v = _to_float(value)
    if v is None or math.isnan(v):
        return None
    return v


def _encode_table(rows: list[list[str]], inputs: list[int]) -> dict[int, dict[str, int]]:
    """Per input column, a code table for the non-numeric training cells."""
    tables: dict[int, dict[str, int]] = {}
    for j in inputs:
        cats: dict[str, int] = {}
        for r in rows:
            if _num(r[j]) is None:
                cats.setdefault(r[j], len(cats))
        tables[j] = cats
    return tables


def _cell(value, j: int, tables: dict[int, dict[str, int]]) -> float:
    """Encode one input cell the same way for training rows and queries."""
    v = _num(value)
    if v is not None:
        return v
    table = tables[j]
    text = "" if value is None else str(value)
    return float(table.get(text, len(table)))


def _scale(vec: list[float], lo: list[float], hi: list[float]) -> list[float]:
    return [
        (v - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0.0
        for k, v in enumerate(vec)
    ]


def _quantile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _model(train_csv) -> dict:
    key = os.fspath(train_csv)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    header, rows = _read(key)
    inputs, goals = _classify(header)
    tables = _encode_table(rows, inputs)
    X = [[_cell(r[j], j, tables) for j in inputs] for r in rows]
    m = len(inputs)
    lo = [min(x[k] for x in X) for k in range(m)] if X else []
    hi = [max(x[k] for x in X) for k in range(m)] if X else []
    Z = [_scale(x, lo, hi) for x in X]

    model = {
        "input_names": [header[j] for j in inputs],
        "inputs": inputs,
        "tables": tables,
        "lo": lo,
        "hi": hi,
        "Z": Z,
        "goal_names": [header[j] for j, _ in goals],
        "goal_vals": [[_num(r[j]) for r in rows] for j, _ in goals],
    }
    _cache[key] = model
    return model


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of an unbenchmarked row.

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns {goal: (estimate, low, high)}, where low..high is the 10th..90th
    percentile of the nearest training rows' values for that goal. A goal
    with no usable neighbour values gets (nan, nan, nan).
    """
    model = _model(train_csv)
    tables = model["tables"]
    qvec = [
        _cell(row.get(name, ""), j, tables)
        for name, j in zip(model["input_names"], model["inputs"])
    ]
    q = _scale(qvec, model["lo"], model["hi"])

    order = sorted((math.dist(q, z), i) for i, z in enumerate(model["Z"]))

    out: dict[str, tuple[float, float, float]] = {}
    for name, values in zip(model["goal_names"], model["goal_vals"]):
        picked: list[float] = []
        for _, i in order:
            v = values[i]
            if v is not None:
                picked.append(v)
                if len(picked) == _K:
                    break
        if not picked:
            out[name] = (math.nan, math.nan, math.nan)
            continue
        s = sorted(picked)
        out[name] = (
            _quantile(s, 0.5),
            _quantile(s, _LO_Q),
            _quantile(s, _HI_Q),
        )
    return out

