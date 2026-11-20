"""step4.py: streaming flow over nightly benchmark rows.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers),
step2 (the training model: encoded inputs, goal values, scaling) and step3
(the significance test).

stream(): a row is kept only if no training row and no earlier row is at
least as good on every goal, using the same normalisation as step1 (1 is
best, scaled by the training ranges). So a kept row is a new best on at
least one goal.

drift(): each new row's distance to its nearest training row, in the scaled
input space from step2, is collected in blocks of _BLOCK rows. Each block is
tested against the leave-one-out nearest-neighbour distances of the training
rows with step3.same. The index of the last row in the first differing block
is returned, or None if no block differs.
"""
from __future__ import annotations

import math
import os
import random

from step1 import _classify, _read
from step2 import _cell, _model, _num, _scale
from step3 import same

_SEED = 1
_BLOCK = 20         # new rows per drift test
_REF_SAMPLE = 200   # training rows used for the reference distances

_cache: dict[str, dict] = {}


def _raw(row, name: str, j: int):
    """Cell of a row given as a dict (by name) or a list (by header index)."""
    if isinstance(row, dict):
        return row.get(name)
    return row[j] if j < len(row) else None


def _ge(a, b) -> bool:
    """True if a is at least as good as b on every goal."""
    return all(x >= y for x, y in zip(a, b))


def _score(vals, ctx) -> tuple[float, ...]:
    """Normalise goal values so that 1 is best, using training ranges.

    Missing values score 0, as in step1. Values outside the training range
    are not clipped, so a row better than anything seen in training scores
    above 1 (or below 0 for the wrong side).
    """
    out: list[float] = []
    for v, lo, hi, sign in zip(vals, ctx["glo"], ctx["ghi"], ctx["signs"]):
        if v is None or lo is None:
            out.append(0.0)
        elif hi == lo:
            out.append(1.0)
        elif sign > 0:
            out.append((v - lo) / (hi - lo))
        else:
            out.append((hi - v) / (hi - lo))
    return tuple(out)


def _context(train_csv) -> dict:
    key = os.fspath(train_csv)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    model = _model(key)
    header, _ = _read(key)
    _, goals = _classify(header)
    goal_vals = model["goal_vals"]  # one list per goal, aligned with train rows

    glo: list[float | None] = []
    ghi: list[float | None] = []
    for col in goal_vals:
        ok = [v for v in col if v is not None]
        glo.append(min(ok) if ok else None)
        ghi.append(max(ok) if ok else None)

    ctx: dict = {
        "input_names": model["input_names"],
        "inputs": model["inputs"],
        "tables": model["tables"],
        "lo": model["lo"],
        "hi": model["hi"],
        "Z": model["Z"],
        "goal_names": model["goal_names"],
        "goal_idx": [j for j, _ in goals],
        "signs": [sign for _, sign in goals],
        "glo": glo,
        "ghi": ghi,
    }

    # Pareto front of the training rows (distinct score vectors only).
    n = len(model["Z"])
    train_scores = {
        _score([col[i] for col in goal_vals], ctx) for i in range(n)
    }
    uniq = sorted(train_scores)
    ctx["front"] = [
        v for v in uniq if not any(w != v and _ge(w, v) for w in uniq)
    ]

    # Reference distribution for drift: leave-one-out nearest-neighbour
    # distances of a sample of training rows, in scaled input space.
    Z = model["Z"]
    ref: list[float] = []
    if len(Z) >= 2:
        rng = random.Random(_SEED)
        idx = sorted(rng.sample(range(len(Z)), min(len(Z), _REF_SAMPLE)))
        for i in idx:
            zi = Z[i]
            ref.append(min(
                math.dist(zi, Z[j]) for j in range(len(Z)) if j != i
            ))
    ctx["ref"] = ref

    _cache[key] = ctx
    return ctx


def _row_scores(row, ctx) -> tuple[float, ...]:
    vals = [
        _num(_raw(row, name, j))
        for name, j in zip(ctx["goal_names"], ctx["goal_idx"])
    ]
    return _score(vals, ctx)


def _row_distance(row, ctx) -> float:
    """Distance from a row's inputs to its nearest training row."""
    tables = ctx["tables"]
    qvec = [
        _cell(_raw(row, name, j), j, tables)
        for name, j in zip(ctx["input_names"], ctx["inputs"])
    ]
    q = _scale(qvec, ctx["lo"], ctx["hi"])
    return min(math.dist(q, z) for z in ctx["Z"])


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or drop (False) each row, in arrival order.

    A row is kept if it is not weakly dominated by the Pareto front of the
    training rows plus all earlier rows. Each decision uses only train_csv
    and the rows before it.
    """
    ctx = _context(train_csv)
    front: list[tuple[float, ...]] = list(ctx["front"])
    keep: list[bool] = []
    for row in rows:
        s = _row_scores(row, ctx)
        if any(_ge(f, s) for f in front):
            keep.append(False)
            continue
        keep.append(True)
        front = [f for f in front if not _ge(s, f)]
        front.append(s)
    return keep


def drift(train_csv, rows) -> int | None:
    """Index into rows where new results first stop looking like train_csv.

    Rows are tested in blocks of _BLOCK. The index returned is the last row
    of the first block whose nearest-neighbour distances differ from the
    training reference (step3.same returns False). None if no block differs.
    A final partial block is not tested.
    """
    ctx = _context(train_csv)
    ref = ctx["ref"]
    if not ref:
        return None
    block: list[float] = []
    for i, row in enumerate(rows):
        block.append(_row_distance(row, ctx))
        if len(block) == _BLOCK:
            if not same(block, ref):
                return i
            block = []
    return None

