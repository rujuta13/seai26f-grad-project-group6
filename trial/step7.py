import csv
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load(blind_csv):
    with open(blind_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _average_ranks(values):
    """Ranks 1..N with ties replaced by their average rank."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[inverse]


def _min_value(raw, sign):
    """Minimisation value: sign is +1 for '-' goals and -1 for '+' goals.
    Missing or non-finite values count as worst (+inf)."""
    v = _to_float(raw)
    if not math.isfinite(v):
        return math.inf
    return sign * v


def _composite(M):
    """Mean normalised rank per row of a minimisation matrix (0 = best, 1 = worst)."""
    m = M.shape[0]
    b = np.zeros(m, dtype=float)
    for j in range(M.shape[1]):
        b += (_average_ranks(M[:, j]) - 1.0) / max(m - 1, 1)
    return b / M.shape[1]


def _pick_next(X, labels, unlabeled, n_goals, rng):
    """Choose the next unlabeled row to benchmark, using a random-forest surrogate
    per goal (lower confidence bound on the normalised goal mean)."""
    idx = np.array(sorted(labels), dtype=int)
    if len(idx) < 4 or len(unlabeled) == 0:
        return int(rng.choice(unlabeled))

    Y = np.array([labels[i] for i in idx], dtype=float)
    Xl = X[idx]
    Xu = X[unlabeled]

    mean_acc = np.zeros(len(unlabeled), dtype=float)
    std_acc = np.zeros(len(unlabeled), dtype=float)
    used = 0
    for j in range(n_goals):
        y = Y[:, j]
        ok = np.isfinite(y)
        if int(ok.sum()) < 4:
            continue
        lo = float(y[ok].min())
        hi = float(y[ok].max())
        span = hi - lo if hi > lo else 1.0
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=1,
            random_state=int(rng.integers(1 << 31)),
        )
        model.fit(Xl[ok], y[ok])
        preds = np.stack([t.predict(Xu) for t in model.estimators_])
        mean_acc += (preds.mean(axis=0) - lo) / span
        std_acc += preds.std(axis=0) / span
        used += 1

    if used == 0:
        return int(rng.choice(unlabeled))

    mean_acc /= used
    std_acc /= used
    score = mean_acc - 1.0 * std_acc
    return int(unlabeled[int(np.argmin(score))])


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, calling label(i) at most `budget` times.

    blind_csv has every goal cell set to '?'. label(i) returns {goal: value} for
    row i. If `goal` names one goal, only that goal is optimised; otherwise all
    goals are optimised together (mean normalised rank, lower is better).
    """
    cols, rows = _load(blind_csv)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no data rows")
    if budget < 1:
        raise ValueError("budget must be at least 1")

    goal_max = [c for c in cols if c.endswith("+")]
    goal_min = [c for c in cols if c.endswith("-")]
    goals_all = goal_min + goal_max
    if goal is None:
        goals = goals_all
    else:
        if goal not in goals_all:
            raise ValueError(f"unknown goal {goal!r}")
        goals = [goal]
    if not goals:
        raise ValueError("no goal columns in blind_csv")
    signs = [1.0 if g.endswith("-") else -1.0 for g in goals]

    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]
    if inputs:
        X = np.array(
            [[_to_float(r.get(c)) for c in inputs] for r in rows], dtype=float
        ).reshape(n, len(inputs))
        X = np.where(np.isnan(X), 0.0, X)
    else:
        X = np.zeros((n, 1), dtype=float)

    labels = {}  # row index -> list of minimisation values, one per optimised goal
    calls = 0

    def evaluate(i):
        nonlocal calls
        if i in labels:
            return
        if calls >= budget:
            raise RuntimeError("label budget exhausted")
        calls += 1
        result = label(i)
        if result is None:
            result = {}
        labels[i] = [_min_value(result.get(g), s) for g, s in zip(goals, signs)]

    rng = np.random.default_rng(0)

    # Initial design: a random sample of rows.
    order = rng.permutation(n)
    n_init = min(budget, n, max(5, budget // 3))
    for i in order[:n_init]:
        evaluate(int(i))

    # Sequential model-guided search with the remaining budget.
    while calls < budget and len(labels) < n:
        unlabeled = np.array([i for i in range(n) if i not in labels], dtype=int)
        nxt = _pick_next(X, labels, unlabeled, len(goals), rng)
        evaluate(nxt)

    idx = np.array(sorted(labels), dtype=int)
    Y = np.array([labels[i] for i in idx], dtype=float)
    comp = _composite(Y)
    best_pos = int(np.argmin(comp))  # ties resolve to the lowest row index
    return int(idx[best_pos])

