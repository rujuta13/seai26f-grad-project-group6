"""Step 14: flow. Streaming keep decisions and drift warnings for scouting reports.

stream(train_csv, rows): a row is kept (True) when no training row and no earlier
row beats it on every goal, i.e. it lies on the Pareto front of the training rows
plus the rows seen so far. Rows with an unusable goal cell are not kept. Each
decision uses only train_csv and the rows before it.

drift(train_csv, rows): warns (returns the index) at the first window of WINDOW
rows where either
  - most window rows are novel in input space (their nearest training row is
    farther away than almost all training rows are from their own neighbour), or
  - a goal's window mean sits far outside the training distribution
    (|z| > GOAL_Z, using the training standard deviation).
Returns None if no warning is raised.
"""
import math

from step1 import _load
from step2 import _column_kinds, _medians, _vector

WINDOW = 15        # rows judged together for drift
NOVEL_FRAC = 0.5   # share of window rows that must look novel to warn on inputs
GOAL_Z = 4.0       # |z| of a goal's window mean that counts as drift
CALIB = 200        # training rows used to calibrate input novelty
NOVEL_Q = 0.95     # quantile of training leave-one-out distances defining "normal"


def _sq(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _oriented(row, goals):
    """Goal vector with higher always better, or None if any goal cell is unusable."""
    out = []
    for c, sign in goals:
        try:
            x = float(row[c])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        out.append(sign * x)
    return out


def _dominated(p, points):
    """True if some point in `points` is at least as good as p on every goal and
    better on at least one."""
    for q in points:
        if all(qk >= pk for qk, pk in zip(q, p)) and any(qk > pk for qk, pk in zip(q, p)):
            return True
    return False


def _nearest_sq(x, pool, skip=-1):
    best = math.inf
    for j, y in enumerate(pool):
        if j != skip:
            d = _sq(x, y)
            if d < best:
                best = d
    return best


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) a row if no training row or earlier row beats it on every goal."""
    train_rows, _, goals = _load(train_csv)
    rows = list(rows)
    if not goals:
        return [False] * len(rows)

    reference = []
    for r in train_rows:
        p = _oriented(r, goals)
        if p is not None:
            reference.append(p)

    seen = []
    keep = []
    for row in rows:
        p = _oriented(row, goals)
        if p is None:
            keep.append(False)
            continue
        keep.append(not (_dominated(p, reference) or _dominated(p, seen)))
        seen.append(p)
    return keep


def drift(train_csv, rows) -> int | None:
    """Index into `rows` where the rows first stop looking like train_csv, or None."""
    train_rows, inputs, goals = _load(train_csv)
    rows = list(rows)
    n = len(train_rows)
    m = len(rows)
    if n == 0 or m == 0:
        return None

    # Input novelty: a query is novel if its nearest training row is farther away
    # than NOVEL_Q of the training rows are from their own nearest neighbour.
    kinds, cats = _column_kinds(train_rows, inputs)
    med = _medians(train_rows, inputs, kinds)
    X = [_vector(r, inputs, kinds, cats, med) for r in train_rows]
    Q = [_vector(r, inputs, kinds, cats, med) for r in rows]

    input_check = bool(X[0]) and n >= 2
    novel = [False] * m
    if input_check:
        step = max(1, n // CALIB)
        loo = sorted(_nearest_sq(X[i], X, skip=i) for i in range(0, n, step))
        thresh = loo[min(len(loo) - 1, int(NOVEL_Q * (len(loo) - 1)))]
        novel = [_nearest_sq(q, X) > thresh for q in Q]

    # Goal statistics from training.
    names = [c for c, _ in goals]
    stats = {}
    for c in names:
        ys = []
        for r in train_rows:
            try:
                y = float(r[c])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(y):
                ys.append(y)
        if not ys:
            continue
        k = len(ys)
        mu = sum(ys) / k
        sd = math.sqrt(sum((y - mu) ** 2 for y in ys) / (k - 1)) if k > 1 else 0.0
        stats[c] = (mu, sd)

    # Per-row goal values of the new rows (unusable cells are skipped).
    qvals = []
    for r in rows:
        vals = {}
        for c in names:
            try:
                y = float(r[c])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(y):
                vals[c] = y
        qvals.append(vals)

    for i in range(WINDOW - 1, m):
        win = range(i - WINDOW + 1, i + 1)

        if input_check and sum(novel[k] for k in win) >= NOVEL_FRAC * WINDOW:
            return i

        for c in names:
            if c not in stats:
                continue
            ws = [qvals[k][c] for k in win if c in qvals[k]]
            if len(ws) < 3:
                continue
            mu, sd = stats[c]
            wm = sum(ws) / len(ws)
            if sd > 0:
                z = (wm - mu) / (sd / math.sqrt(len(ws)))
                if abs(z) > GOAL_Z:
                    return i
            elif abs(wm - mu) > 1e-9 * max(1.0, abs(mu)):
                return i

    return None

