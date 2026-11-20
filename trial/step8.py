"""step8.py: fix a bad build with as few input changes as possible.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers),
step2 (training model and kNN goal estimates) and step4 (goal scoring
against the training ranges).

A build is "good" when its mean normalised goal score is at least the
training median, the same threshold step6.why uses. The goals of a candidate
build are unknown, so each candidate is judged by the kNN estimate from
step2.guess. fix() searches candidates in order of how many inputs they
change (0, then 1, then 2, ...). It returns the first size that yields a good
candidate, choosing the best predicted score at that size. Columns named in
`frozen` are never changed.
"""
from __future__ import annotations

import itertools
import math
import os
import statistics

from step1 import _classify, _read, _to_float
from step2 import _model, guess
from step4 import _context, _score

_MAX_CHANGES = 4   # largest number of inputs a fix may change
_MAX_EVALS = 5000  # cap on kNN estimates per call
_MAX_LEVELS = 10   # candidate values per column (evenly spaced if more)


def _typed(text: str):
    """A cell as a float if numeric, else as a stripped string."""
    s = text.strip()
    v = _to_float(s)
    if v is None or math.isnan(v):
        return s
    return v


def _norm(value):
    """Comparable form of an input value. Missing cells become ''."""
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        return "" if math.isnan(v) else v
    return _typed(str(value))


def _levels(rows: list[list[str]], j: int) -> list:
    """Distinct training values of input column j (evenly spaced if many)."""
    seen: dict = {}
    for r in rows:
        v = _norm(r[j])
        if v != "":
            seen[v] = True
    ordered = sorted(seen, key=lambda v: (isinstance(v, str), v))
    if len(ordered) > _MAX_LEVELS:
        last = len(ordered) - 1
        keep = sorted({
            round(b * last / (_MAX_LEVELS - 1)) for b in range(_MAX_LEVELS)
        })
        ordered = [ordered[i] for i in keep]
    return ordered


def _threshold(ctx: dict, goal_vals: list[list], n: int) -> float:
    """Median over training rows of the mean normalised goal score."""
    means: list[float] = []
    for i in range(n):
        s = _score([col[i] for col in goal_vals], ctx)
        means.append(sum(s) / len(s))
    return statistics.median(means)


def _predict(key: str, cand: dict, ctx: dict) -> float:
    """Mean normalised score of a candidate, using kNN goal estimates."""
    est = guess(key, cand)
    vals: list[float | None] = []
    for name in ctx["goal_names"]:
        e = est[name][0]
        vals.append(None if math.isnan(e) else e)
    s = _score(vals, ctx)
    return sum(s) / len(s)


def _changes(start: tuple, free: list[tuple[int, list]], k: int):
    """Candidate value tuples that change exactly k free columns."""
    for cols in itertools.combinations(range(len(free)), k):
        for picks in itertools.product(*(free[c][1] for c in cols)):
            values = list(start)
            for c, v in zip(cols, picks):
                values[free[c][0]] = v
            yield tuple(values)


def _result(row: dict, names: list[str], start: tuple, values: tuple) -> dict:
    """The original row with changed inputs replaced by training values."""
    out = dict(row)
    for name, before, after in zip(names, start, values):
        if after != before:
            out[name] = after
    return out


def fix(train_csv, row, frozen=()) -> dict:
    """Same inputs as `row`, with as few values changed as possible so that
    the build becomes good. Columns named in `frozen` are never changed.

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns a dict with the same keys as `row`, where changed cells hold
    values seen in the training file. If the row is already predicted good,
    it is returned unchanged. If no candidate with at most _MAX_CHANGES
    changes is predicted good (or the evaluation budget runs out), the
    candidate with the highest predicted score is returned.
    """
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    if not rows or not goals:
        raise ValueError("training file needs rows and goal columns")
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)

    names = [header[j] for j in inputs]
    ctx = _context(key)
    threshold = _threshold(ctx, _model(key)["goal_vals"], len(rows))

    start = tuple(_norm(row.get(name)) for name in names)
    free: list[tuple[int, list]] = []
    for pos, (name, j) in enumerate(zip(names, inputs)):
        if name in frozen:
            continue
        alts = [v for v in _levels(rows, j) if v != start[pos]]
        if alts:
            free.append((pos, alts))

    memo: dict[tuple, float] = {}
    spent = 0

    def predicted(values: tuple) -> float | None:
        nonlocal spent
        if values in memo:
            return memo[values]
        if spent >= _MAX_EVALS:
            return None
        spent += 1
        s = _predict(key, dict(zip(names, values)), ctx)
        memo[values] = s
        return s

    best_score = predicted(start)
    assert best_score is not None
    if best_score >= threshold:
        return _result(row, names, start, start)

    best_values = start
    for k in range(1, min(_MAX_CHANGES, len(free)) + 1):
        level_good: tuple[float, tuple] | None = None
        out_of_budget = False
        for values in _changes(start, free, k):
            s = predicted(values)
            if s is None:
                out_of_budget = True
                break
            if s >= threshold and (level_good is None or s > level_good[0]):
                level_good = (s, values)
            if s > best_score:
                best_score, best_values = s, values
        if level_good is not None:
            return _result(row, names, start, level_good[1])
        if out_of_budget:
            break
    return _result(row, names, start, best_values)

