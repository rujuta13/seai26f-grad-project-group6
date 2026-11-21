"""step20.py: race our shortlist and penalty guesser against simple rivals.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers,
goal scoring, input encoding and the 5% cap), step2 (kNN goal guesses),
step4 (training ranges and goal scoring) and step10 (ridge helpers).

Shortlist race (higher is better). Each method picks k test rows using only
their input columns, then is scored by the mean normalised goal score of the
picked rows (step1 scaling over the test file, 1 is best). k is the 5% cap
from step1, so every method gets the same shortlist size.
  - ours: step2.guess predicts each test row's goals from its inputs, the
    prediction is scored against the training ranges (step4), and the k best
    predictions form the shortlist.
  - top_overall: the k rows with the highest OVR, the simple rating list.
  - oracle: the k rows with the best true score, the ceiling.

Penalty guesser race (higher is better): negative mean absolute error of
Penalties+ on the test rows, in goal units. 0 is perfect.
  - guess_ours: the step2.guess median for Penalties+. With no usable
    neighbour value, the training mean of Penalties+ is used instead.
  - guess_ridge: ridge regression on the input flags from step10, the
    off-the-shelf rival.
"""
from __future__ import annotations

import math
import os

from step1 import (
    _BEST_DEN,
    _BEST_NUM,
    _classify,
    _encode_inputs,
    _goal_scores,
    _minmax,
    _read,
    _to_float,
)
from step2 import guess
from step4 import _context, _score
from step10 import _RIDGE, _at, _dot, _ridge_fit

_OVR = "OVR"
_PEN_PREFIX = "Penalties"


def _goal_index(header: list[str], goals, prefix: str) -> int | None:
    """Column index of the first goal whose name starts with prefix."""
    for j, _ in goals:
        if header[j].startswith(prefix):
            return j
    return None


def _top_k(values: list[float], k: int) -> list[int]:
    """Indices of the k largest values (ties keep file order)."""
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def _mean_over(values: list[float], idx: list[int]) -> float:
    return sum(values[i] for i in idx) / len(idx)


def _test_inputs(row: list[str], names: list[str],
                 te_cols: list[int | None]) -> dict[str, str]:
    """Input cells of a test row, keyed by training input column name."""
    return {
        nm: (_at(row, j) if j is not None else "")
        for nm, j in zip(names, te_cols)
    }


def _predicted_scores(tr_path: str, names: list[str],
                      te_cols: list[int | None],
                      rows: list[list[str]], ctx: dict) -> list[float]:
    """Mean normalised score of each test row, from kNN goal estimates."""
    out: list[float] = []
    for r in rows:
        est = guess(tr_path, _test_inputs(r, names, te_cols))
        vals: list[float | None] = []
        for name in ctx["goal_names"]:
            e = est.get(name, (math.nan, math.nan, math.nan))[0]
            vals.append(None if math.isnan(e) else e)
        s = _score(vals, ctx)
        out.append(sum(s) / len(s))
    return out


def _penalty_scores(tr_path: str, train_header: list[str],
                    train_rows: list[list[str]],
                    rows: list[list[str]],
                    tr_pen: int, te_pen: int,
                    names: list[str],
                    te_cols: list[int | None]) -> dict[str, float]:
    """Negative MAE of Penalties+ for step2.guess and the ridge rival."""
    if not names:
        return {}
    tr_inputs, _ = _classify(train_header)
    tr_proj = [[_at(r, j) for j in tr_inputs] for r in train_rows]
    te_proj = [
        [_at(r, j) if j is not None else "" for j in te_cols]
        for r in rows
    ]
    enc = _minmax(_encode_inputs(tr_proj + te_proj, list(range(len(names)))))
    m = len(tr_proj)
    Xtr, Xte = enc[:m], enc[m:]

    fit_x: list[list[float]] = []
    fit_y: list[float] = []
    for x, r in zip(Xtr, train_rows):
        v = _to_float(_at(r, tr_pen))
        if v is not None and math.isfinite(v):
            fit_x.append([1.0] + x)
            fit_y.append(v)
    if not fit_x:
        return {}
    beta = _ridge_fit(fit_x, fit_y, _RIDGE)
    fallback = sum(fit_y) / len(fit_y)
    pen_name = train_header[tr_pen]

    ridge_err: list[float] = []
    guess_err: list[float] = []
    for x, r in zip(Xte, rows):
        truth = _to_float(_at(r, te_pen))
        if truth is None or not math.isfinite(truth):
            continue

        ridge_err.append(abs(_dot(beta, [1.0] + x) - truth))

        est = guess(tr_path, _test_inputs(r, names, te_cols)).get(
            pen_name, (math.nan, math.nan, math.nan))[0]
        if math.isfinite(est):
            guess_err.append(abs(est - truth))
        else:
            guess_err.append(abs(fallback - truth))

    def neg_mean(errs: list[float]) -> float:
        return -sum(errs) / len(errs) if errs else -math.inf

    return {
        "guess_ours": neg_mean(guess_err),
        "guess_ridge": neg_mean(ridge_err),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores of ours and the rivals on test_csv (higher is better).

    Shortlist keys: "ours", "top_overall", and "oracle" (the best possible
    shortlist of the same size). Penalty keys, when both files have a
    Penalties+ goal and inputs: "guess_ours" and "guess_ridge".
    """
    tr_path = os.fspath(train_csv)
    te_path = os.fspath(test_csv)
    train_header, train_rows = _read(tr_path)
    header, rows = _read(te_path)
    _, tr_goals = _classify(train_header)
    _, goals = _classify(header)
    if not rows or not goals:
        raise ValueError("test_csv needs rows and goal columns")
    if not train_rows or not tr_goals:
        raise ValueError("train_csv needs rows and goal columns")

    n = len(rows)
    k = max(1, n * _BEST_NUM // _BEST_DEN)
    truth = [sum(s) / len(s) for s in _goal_scores(rows, goals)]

    results: dict[str, float] = {
        "oracle": _mean_over(truth, _top_k(truth, k)),
    }

    ctx = _context(tr_path)
    tr_inputs, _ = _classify(train_header)
    names = [train_header[j] for j in tr_inputs]
    te_index = {nm: j for j, nm in enumerate(header)}
    te_cols: list[int | None] = [te_index.get(nm) for nm in names]

    pred = _predicted_scores(tr_path, names, te_cols, rows, ctx)
    results["ours"] = _mean_over(truth, _top_k(pred, k))

    ovr_j = te_index.get(_OVR)
    if ovr_j is not None:
        ovr: list[float] = []
        for r in rows:
            v = _to_float(_at(r, ovr_j))
            ovr.append(-math.inf if v is None or math.isnan(v) else v)
        results["top_overall"] = _mean_over(truth, _top_k(ovr, k))

    tr_pen = _goal_index(train_header, tr_goals, _PEN_PREFIX)
    te_pen = _goal_index(header, goals, _PEN_PREFIX)
    if tr_pen is not None and te_pen is not None:
        results.update(_penalty_scores(
            tr_path, train_header, train_rows, rows,
            tr_pen, te_pen, names, te_cols))

    return results

