"""step7.py: choose a good flag set while measuring at most `budget` builds.

The blind file has every goal cell set to '?', so only the input columns can be
compared before measuring. choose() spends the budget in two phases. The first
half spreads measurements across the input space (farthest-point sampling from
the centre). The second half alternates between the nearest unmeasured build to
the current best and the build farthest from everything measured so far. The
answer is the best measured build: the Pareto front of the measured builds, then
the lowest mean goal rank on that front.

Loading, column classification, input encoding and goal matrices come from step1.
The mean-rank helper comes from step6.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _goal_matrix, _input_matrix, _load
from step6 import _mean_rank

_EXPLORE_SHARE = 0.5   # share of the budget spent on spread-out sampling before exploiting


def _farthest(D, chosen, pool):
    """Member of `pool` whose distance to its nearest member of `chosen` is largest."""
    pool = np.asarray(pool, dtype=int)
    chosen = np.asarray(chosen, dtype=int)
    dmin = D[np.ix_(pool, chosen)].min(axis=1)
    return int(pool[int(np.argmax(dmin))])


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, after calling label(i) at most `budget` times.

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column name: value}.
    goal: optional name of one goal column; if given, only that goal is optimised.
    """
    if isinstance(budget, bool) or not float(budget).is_integer() or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)

    header, rows = _load(blind_csv)
    if not rows:
        raise ValueError("blind_csv contains no data rows")
    n = len(rows)

    maximise, minimise, inputs = _classify(header)
    if goal is not None:
        if goal not in header:
            raise ValueError(f"unknown goal column: {goal!r}")
        col = header.index(goal)
        if goal.endswith("+"):
            maximise, minimise = [col], []
        elif goal.endswith("-"):
            maximise, minimise = [], [col]
        else:
            raise ValueError("goal must name a column ending in '+' or '-'")
    goal_cols = minimise + maximise
    if not goal_cols:
        raise ValueError("blind_csv has no goal columns")

    # Standardised input space, used only to decide where to measure next.
    X = _input_matrix(rows, inputs)
    if X.shape[1] > 0:
        Z = StandardScaler().fit_transform(X)
    else:
        Z = np.zeros((n, 0), dtype=float)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))

    limit = min(budget, n)
    measured: dict[int, dict[int, float]] = {}

    def measure(i):
        if len(measured) >= budget:
            raise RuntimeError("measurement budget exceeded")
        vals = label(i)
        measured[i] = {c: float(vals[header[c]]) for c in goal_cols}

    def pick():
        """Best measured row: Pareto front first, then lowest mean rank on that front."""
        ids = sorted(measured)
        raw = []
        for i in ids:
            cells = list(rows[i])
            for c in goal_cols:
                cells[c] = repr(measured[i][c])
            raw.append(cells)
        G = _goal_matrix(raw, maximise, minimise)
        le = (G[None, :, :] <= G[:, None, :]).all(axis=2)   # row j no worse than row i everywhere
        lt = (G[None, :, :] < G[:, None, :]).any(axis=2)    # and strictly better somewhere
        front = np.flatnonzero(~(le & lt).any(axis=1))
        score = _mean_rank(G[front])
        return ids[int(front[int(np.argmin(score))])]

    # Phase 1: start from the build closest to the centre, then spread out.
    centre = Z.mean(axis=0) if Z.shape[1] > 0 else np.zeros(0, dtype=float)
    first = int(np.argmin(((Z - centre) ** 2).sum(axis=1)))
    measure(first)
    explore_target = min(limit, max(1, math.ceil(limit * _EXPLORE_SHARE)))
    while len(measured) < explore_target:
        unmeasured = [i for i in range(n) if i not in measured]
        measure(_farthest(D, list(measured), unmeasured))

    # Phase 2: alternate between refining near the best and exploring the gaps.
    step = 0
    while len(measured) < limit:
        unmeasured = [i for i in range(n) if i not in measured]
        if not unmeasured:
            break
        if step % 2 == 0:
            anchor = pick()
            cand = min(unmeasured, key=lambda j: (float(D[anchor, j]), j))
        else:
            cand = _farthest(D, list(measured), unmeasured)
        measure(cand)
        step += 1

    return pick()

