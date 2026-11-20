"""step10.py: race our tuner and guesser against standard rivals.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers,
goal scoring, input encoding), step2 (guess) and step7 (choose).

The standard library has no hyperparameter tuner and no regression model,
so the rivals are written here with the standard library only:
  - random: uniform random search with the same label budget as ours.
  - hillclimb: greedy nearest-neighbour local search with random restarts,
    standing in for a generic tuner.
  - ridge: ridge regression on the input flags predicting log build speed,
    solved by normal equations in pure Python. It is the off-the-shelf-style
    rival for the guesser.

Tuner scores: mean normalised goal score of the build each method picks
(step1 scaling over the test file, 1 is best). "oracle" is the best test
build, the ceiling. Higher is better.

Guesser scores: negative mean absolute percentage error of build speed
(the goal whose name starts with "Performance") on the test rows. 0 is
perfect, so higher is better. "guess_ours" uses step2.guess and
"guess_ridge" uses the ridge rival.
"""
from __future__ import annotations

import csv
import math
import os
import random
import tempfile

from step1 import (
    _classify,
    _encode_inputs,
    _goal_scores,
    _minmax,
    _read,
    _to_float,
)
from step2 import guess
from step7 import choose

_BUDGET = 30          # label calls allowed to every tuner
_SEED = 1
_STALE_LIMIT = 3      # non-improving hill-climb probes before a restart
_RIDGE = 1e-3         # ridge penalty on the input weights
_SPEED_PREFIX = "Performance"
_EXP_CAP = 700.0      # keeps math.exp finite


def _at(row: list[str], j: int) -> str:
    """Cell j of a row, or '' if the row is short."""
    return row[j] if j < len(row) else ""


def _label_fn(header: list[str], rows: list[list[str]], goals):
    """label(i): goal values of test row i as {goal column name: float}."""
    def label(i: int) -> dict[str, float]:
        r = rows[i]
        out: dict[str, float] = {}
        for j, _ in goals:
            v = _to_float(_at(r, j))
            out[header[j]] = math.nan if v is None else v
        return out
    return label


def _write_blind(header: list[str], rows: list[list[str]],
                 goal_cols: set[int], dest: str) -> None:
    """Write a copy of the test file with every goal cell replaced by '?'."""
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                "?" if j in goal_cols else _at(r, j)
                for j in range(len(header))
            ])


def _budgeted(label, budget: int):
    """Return (probe, seen). probe(i) calls label once per new row."""
    seen: dict[int, dict[str, float]] = {}

    def probe(i: int) -> None:
        if i in seen:
            return
        if len(seen) >= budget:
            raise RuntimeError(f"label budget of {budget} calls exhausted")
        seen[i] = label(i)

    return probe, seen


def _observed_scores(header: list[str], goals, seen) -> dict[int, float]:
    """Mean normalised score of each probed row, normalised over probed rows."""
    ids = sorted(seen)
    table: list[list[str]] = []
    for i in ids:
        row = ["?"] * len(header)
        for j, _ in goals:
            row[j] = str(seen[i][header[j]])
        table.append(row)
    S = _goal_scores(table, goals)
    return {i: sum(s) / len(s) for i, s in zip(ids, S)}


def _best_of(header: list[str], goals, seen) -> int:
    sc = _observed_scores(header, goals, seen)
    return max(sc, key=lambda i: (sc[i], -i))


def _hillclimb(X, budget: int, probe, seen, rng: random.Random,
               header: list[str], goals) -> None:
    """Greedy nearest-neighbour local search with random restarts.

    Each step probes the unprobed row nearest to the current best. After
    _STALE_LIMIT steps without improvement it restarts at a random row.
    """
    n = len(X)
    probe(rng.randrange(n))
    stale = 0
    while len(seen) < budget:
        unseen = [i for i in range(n) if i not in seen]
        if not unseen:
            break
        sc = _observed_scores(header, goals, seen)
        top = max(sc, key=lambda i: (sc[i], -i))
        if stale >= _STALE_LIMIT:
            nxt = rng.choice(unseen)
            stale = 0
        else:
            nxt = min(unseen, key=lambda i: (math.dist(X[top], X[i]), i))
        probe(nxt)
        sc = _observed_scores(header, goals, seen)
        stale = 0 if sc[nxt] > sc[top] else stale + 1


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    m = len(b)
    M = [A[i][:] + [b[i]] for i in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        if abs(piv) < 1e-12:
            continue
        for r in range(m):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / piv
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [
        M[i][m] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
        for i in range(m)
    ]


def _ridge_fit(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Ridge weights via normal equations. X[0]-column is the intercept."""
    m = len(X[0])
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, t in zip(X, y):
        for p in range(m):
            xp = x[p]
            b[p] += xp * t
            row = A[p]
            for q in range(m):
                row[q] += xp * x[q]
    for p in range(1, m):  # the intercept is not penalised
        A[p][p] += lam
    return _solve(A, b)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(p * q for p, q in zip(a, b))


def _speed_goal(header: list[str], goals) -> str | None:
    """Name of the build-speed goal, if the file has one."""
    for j, _ in goals:
        if header[j].startswith(_SPEED_PREFIX):
            return header[j]
    return None


def _guesser_scores(tr_path: str, train_header: list[str],
                    train_rows: list[list[str]], header: list[str],
                    rows: list[list[str]], speed: str) -> dict[str, float]:
    """Negative MAPE of build speed for step2.guess and the ridge rival."""
    tr_inputs, _ = _classify(train_header)
    names = [train_header[j] for j in tr_inputs]
    if any(nm not in header for nm in names) or speed not in train_header:
        return {}

    te_cols = [header.index(nm) for nm in names]
    k = len(names)
    tr_proj = [[_at(r, j) for j in tr_inputs] for r in train_rows]
    te_proj = [[_at(r, j) for j in te_cols] for r in rows]
    enc = _minmax(_encode_inputs(tr_proj + te_proj, list(range(k))))
    m = len(tr_proj)
    Xtr, Xte = enc[:m], enc[m:]

    tr_j = train_header.index(speed)
    te_j = header.index(speed)

    fit_x: list[list[float]] = []
    fit_y: list[float] = []
    for x, r in zip(Xtr, train_rows):
        v = _to_float(_at(r, tr_j))
        if v is not None and math.isfinite(v) and v > 0:
            fit_x.append([1.0] + x)
            fit_y.append(math.log(v))
    if not fit_x:
        return {}
    beta = _ridge_fit(fit_x, fit_y, _RIDGE)

    ridge_err: list[float] = []
    guess_err: list[float] = []
    for x, r in zip(Xte, rows):
        truth = _to_float(_at(r, te_j))
        if truth is None or not math.isfinite(truth) or truth <= 0:
            continue

        z = min(_dot(beta, [1.0] + x), _EXP_CAP)
        ridge_err.append(abs(math.exp(z) - truth) / truth)

        row_in = {nm: _at(r, j) for nm, j in zip(names, te_cols)}
        est = guess(tr_path, row_in).get(
            speed, (math.nan, math.nan, math.nan))[0]
        if math.isfinite(est):
            guess_err.append(abs(est - truth) / truth)
        else:
            guess_err.append(1.0)  # no usable neighbour: count as 100% off

    def neg_mean(errs: list[float]) -> float:
        return -sum(errs) / len(errs) if errs else -math.inf

    return {
        "guess_ours": neg_mean(guess_err),
        "guess_ridge": neg_mean(ridge_err),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores of ours and the rivals on test_csv (higher is better).

    Tuner keys: "ours", "random", "hillclimb", plus "oracle" (the best
    test build). Guesser keys, when the file has a build-speed goal:
    "guess_ours" and "guess_ridge".
    """
    tr_path = os.fspath(train_csv)
    te_path = os.fspath(test_csv)
    train_header, train_rows = _read(tr_path)
    header, rows = _read(te_path)
    inputs, goals = _classify(header)
    if not rows or not goals:
        raise ValueError("test_csv needs rows and goal columns")

    n = len(rows)
    budget = min(_BUDGET, n)
    goal_cols = {j for j, _ in goals}
    truth = [sum(s) / len(s) for s in _goal_scores(rows, goals)]
    label = _label_fn(header, rows, goals)

    results: dict[str, float] = {"oracle": max(truth)}

    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(header, rows, goal_cols, blind)

        results["ours"] = truth[choose(blind, label, budget=budget)]

        probe, seen = _budgeted(label, budget)
        for i in random.Random(_SEED).sample(range(n), budget):
            probe(i)
        results["random"] = truth[_best_of(header, goals, seen)]

        probe, seen = _budgeted(label, budget)
        X = _minmax(_encode_inputs(rows, inputs))
        _hillclimb(X, budget, probe, seen, random.Random(_SEED),
                   header, goals)
        results["hillclimb"] = truth[_best_of(header, goals, seen)]

    speed = _speed_goal(header, goals)
    if speed is not None:
        results.update(_guesser_scores(
            tr_path, train_header, train_rows, header, rows, speed))

    return results

