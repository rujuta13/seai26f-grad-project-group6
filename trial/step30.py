"""step30.py: race our targeting and response guesser against simple rivals.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers,
goal scoring, input encoding and scaling), step4 (training context), step10
(ridge fit and helpers), step12 (calibrated response estimate) and step20
(predicted-score targeting, top-k helpers and goal lookup).

Targeting race (higher is better). Each method mails a list of k test
customers, where k is the client's mailing budget of 50, capped at the file
size. Each list is scored by the mean normalised goal score of the mailed
customers (step1 scaling over the test file, 1 is best).
  - ours: step20 predicted scores from inputs alone; mail the k customers
    with the best predictions.
  - highest_income: mail the k customers with the highest INCOME.
  - random: mail k random customers (seeded).
  - oracle: the k customers with the best true score, the ceiling.

Response guesser race (higher is better): negative Brier score of the
predicted probability that a customer answers the campaign (the RESPONSE+
goal, 0 or 1) on the test rows. 0 is perfect; predictions are clamped to
[0, 1]. Rows with no usable response value are skipped.
  - guess_ours: step12.guess median for RESPONSE+ (training rate if no
    usable neighbour value).
  - guess_ridge: ridge regression on the input flags, the off-the-shelf
    model rival.
  - guess_mean: the training response rate for every customer.
"""
from __future__ import annotations

import math
import os
import random

from step1 import (
    _classify,
    _encode_inputs,
    _goal_scores,
    _minmax,
    _read,
    _to_float,
)
from step4 import _context
from step10 import _RIDGE, _at, _dot, _ridge_fit
from step12 import guess as _calibrated_guess
from step20 import (
    _goal_index,
    _mean_over,
    _predicted_scores,
    _test_inputs,
    _top_k,
)

_MAIL_BUDGET = 50     # customers the campaign can mail
_SEED = 1
_INCOME = "INCOME"
_RESPONSE = "RESPONSE"


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _neg_mean(vals: list[float]) -> float:
    return -sum(vals) / len(vals) if vals else -math.inf


def _income_values(header: list[str], rows: list[list[str]]) -> list[float] | None:
    """INCOME per test row, with missing values ranked last. None if absent."""
    if _INCOME not in header:
        return None
    j = header.index(_INCOME)
    out: list[float] = []
    for r in rows:
        v = _to_float(_at(r, j))
        out.append(-math.inf if v is None or math.isnan(v) else v)
    return out


def _targeting_race(tr_path: str, header: list[str], rows: list[list[str]],
                    truth: list[float], names: list[str],
                    te_cols: list[int | None], ctx: dict,
                    k: int) -> dict[str, float]:
    """Mean true score of the k mailed customers for each targeting method."""
    results: dict[str, float] = {
        "oracle": _mean_over(truth, _top_k(truth, k)),
    }

    pred = _predicted_scores(tr_path, names, te_cols, rows, ctx)
    results["ours"] = _mean_over(truth, _top_k(pred, k))

    income = _income_values(header, rows)
    if income is not None:
        results["highest_income"] = _mean_over(truth, _top_k(income, k))

    picks = random.Random(_SEED).sample(range(len(rows)), k)
    results["random"] = _mean_over(truth, picks)
    return results


def _response_race(tr_path: str, train_header: list[str],
                   train_rows: list[list[str]], rows: list[list[str]],
                   names: list[str], te_cols: list[int | None],
                   tr_resp: int, te_resp: int) -> dict[str, float]:
    """Negative Brier score of the response guessers on the test rows."""
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
        v = _to_float(_at(r, tr_resp))
        if v is not None and math.isfinite(v):
            fit_x.append([1.0] + x)
            fit_y.append(v)
    if not fit_x:
        return {}
    beta = _ridge_fit(fit_x, fit_y, _RIDGE)
    base_rate = sum(fit_y) / len(fit_y)
    resp_name = train_header[tr_resp]

    sq_ours: list[float] = []
    sq_ridge: list[float] = []
    sq_mean: list[float] = []
    for x, r in zip(Xte, rows):
        y = _to_float(_at(r, te_resp))
        if y is None or not math.isfinite(y):
            continue

        ridge_p = _clamp01(_dot(beta, [1.0] + x))

        est = _calibrated_guess(
            tr_path, _test_inputs(r, names, te_cols)
        ).get(resp_name, (math.nan, math.nan, math.nan))[0]
        ours_p = _clamp01(est) if math.isfinite(est) else base_rate

        sq_ours.append((ours_p - y) ** 2)
        sq_ridge.append((ridge_p - y) ** 2)
        sq_mean.append((base_rate - y) ** 2)

    return {
        "guess_ours": _neg_mean(sq_ours),
        "guess_ridge": _neg_mean(sq_ridge),
        "guess_mean": _neg_mean(sq_mean),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores of ours and the rivals on test_csv (higher is better).

    Targeting keys: "ours", "highest_income", "random", and "oracle" (the
    best possible mailing list of the same size). Response keys, when both
    files have a RESPONSE+ goal: "guess_ours", "guess_ridge", "guess_mean".
    """
    tr_path = os.fspath(train_csv)
    te_path = os.fspath(test_csv)
    train_header, train_rows = _read(tr_path)
    header, rows = _read(te_path)
    tr_inputs, tr_goals = _classify(train_header)
    _, goals = _classify(header)
    if not rows or not goals:
        raise ValueError("test_csv needs rows and goal columns")
    if not train_rows or not tr_goals:
        raise ValueError("train_csv needs rows and goal columns")

    n = len(rows)
    k = min(_MAIL_BUDGET, n)
    truth = [sum(s) / len(s) for s in _goal_scores(rows, goals)]

    ctx = _context(tr_path)
    names = [train_header[j] for j in tr_inputs]
    te_index = {nm: j for j, nm in enumerate(header)}
    te_cols: list[int | None] = [te_index.get(nm) for nm in names]

    results = _targeting_race(tr_path, header, rows, truth, names,
                              te_cols, ctx, k)

    tr_resp = _goal_index(train_header, tr_goals, _RESPONSE)
    if tr_resp is not None:
        te_resp = te_index.get(train_header[tr_resp])
        if te_resp is not None:
            results.update(_response_race(
                tr_path, train_header, train_rows, rows, names, te_cols,
                tr_resp, te_resp))

    return results

