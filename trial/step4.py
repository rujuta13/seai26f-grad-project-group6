"""step4.py: streaming keep/drop decisions and drift warnings for nightly results."""

import numpy as np

from step1 import _load
from step3 import same

# Number of consecutive new rows tested together for drift.
_BLOCK = 15


def _goal_matrix(rows, goals):
    """Matrix over the goals where larger is always better (minimised goals negated)."""
    return np.array(
        [[float(r[name]) * sign for name, sign in goals] for r in rows],
        dtype=float,
    ).reshape(len(rows), len(goals))


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) each incoming row that is not dominated on the goals by any
    row seen so far (train_csv rows plus earlier incoming rows)."""
    train_rows, _, goals = _load(train_csv)
    if not goals:
        return [False] * len(rows)

    ref = _goal_matrix(train_rows, goals)
    keep: list[bool] = []
    for r in rows:
        v = _goal_matrix([r], goals)[0]
        if ref.shape[0] == 0:
            dominated = False
        else:
            ge = (ref >= v).all(axis=1)
            gt = (ref > v).any(axis=1)
            dominated = bool((ge & gt).any())
        keep.append(not dominated)
        # Every row seen so far becomes part of the reference for later rows.
        ref = np.vstack([ref, v[None, :]])
    return keep


def drift(train_csv, rows) -> int | None:
    """Index into `rows` at which the goal values of new rows first stop looking
    like those in train_csv, judged block by block with a two-sided permutation
    test (step3.same). Returns None if no block differs."""
    train_rows, _, goals = _load(train_csv)
    if not goals or not train_rows or not rows:
        return None

    names = [name for name, _ in goals]
    train_vals = {name: [float(r[name]) for r in train_rows] for name in names}

    for start in range(0, len(rows), _BLOCK):
        block = rows[start:start + _BLOCK]
        if len(block) < _BLOCK:
            break
        end = start + len(block) - 1
        for name in names:
            xs = [float(r[name]) for r in block]
            if not same(xs, train_vals[name]):
                return end
    return None

