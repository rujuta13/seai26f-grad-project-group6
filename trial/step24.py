"""Step 24: flow. Streaming welcome-offer picks and sign-up drift warnings.

stream(train_csv, rows): rows arrive one at a time. A new customer is kept (True)
when no training customer and no earlier row beats it on every goal, i.e. it lies
on the Pareto front of the training rows plus the rows seen so far. Rows with an
unusable goal cell are not kept. Each decision uses only train_csv and the rows
before it.

drift(train_csv, rows): returns the index into `rows` of the first row at which
the sign-ups stop looking like train_csv, or None. A warning is raised on a window
of WINDOW rows where either
  - most window rows are novel in input space (their nearest training customer is
    farther away than almost all training customers are from their own nearest
    neighbour), or
  - a goal's window mean sits far outside the training distribution
    (|z| > GOAL_Z, using the training standard deviation).

Inputs are scaled over the training file so that large-valued columns (income,
spending) do not drown out small ones (children at home, campaign flags). Numeric
ranges use the 1st..99th percentiles so a single typo does not squash the scale,
while out-of-range values still count as novel. Identifier-like numeric columns
and text columns with many levels are not used.
"""
import math
import random

from step1 import _load

WINDOW = 15          # rows judged together for drift
NOVEL_FRAC = 0.5     # share of window rows that must look novel to warn on inputs
GOAL_Z = 4.0         # |z| of a goal's window mean that counts as drift
CALIB = 200          # pool rows used to calibrate input novelty
NOVEL_Q = 0.95       # quantile of leave-one-out distances defining "normal"
MAX_POOL = 1000      # training rows used for neighbour search (seeded sample if more)
SEED = 1             # fixed seed so results are reproducible
NUMERIC_FRAC = 0.9   # a column is numeric if this share of its training cells parse
ID_FRAC = 0.9        # numeric column with more distinct values than this share is an id
MAX_LEVELS = 25      # text columns with more levels are identifier-like and not used
CLIP_Q = 0.01        # numeric scaling uses these quantiles as the range
EPS = 1e-12


def _finite(v):
    """The cell as a finite float, or None if it is missing, non-numeric or non-finite."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v.strip()
    return "" if v is None else str(v).strip()


def _quantile(srt, q):
    """Linear-interpolated quantile of a non-empty ascending list."""
    pos = q * (len(srt) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(srt) - 1)
    return srt[lo] + (srt[hi] - srt[lo]) * (pos - lo)


def _dominates(q, p):
    """True if q is at least as good as p on every goal and better on at least one."""
    return all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p))


def _oriented(row, goals):
    """Goal vector with higher always better, or None if any goal cell is unusable."""
    out = []
    for c, sign in goals:
        x = _finite(row.get(c))
        if x is None:
            return None
        out.append(sign * x)
    return out


def _front(points):
    """Pareto front (maximal elements) of a list of oriented goal vectors."""
    order = sorted(points, key=lambda p: -sum(p))
    front = []
    for p in order:
        if any(_dominates(q, p) for q in front):
            continue
        front.append(p)
    return front


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) a row if no training row or earlier row beats it on every goal.

    The maximal set M of all points seen so far (training rows plus earlier rows)
    is maintained. A point is dominated by some seen point exactly when it is
    dominated by a member of M, so M gives the same decisions as checking every
    seen row.
    """
    train_rows, _, goals = _load(train_csv)
    rows = list(rows)
    if not goals:
        return [False] * len(rows)

    reference = []
    for r in train_rows:
        p = _oriented(r, goals)
        if p is not None:
            reference.append(p)
    M = _front(reference)

    keep = []
    for row in rows:
        p = _oriented(row, goals)
        if p is None:
            keep.append(False)
            continue
        if any(_dominates(m, p) for m in M):
            keep.append(False)
            continue
        keep.append(True)
        M = [m for m in M if not _dominates(p, m)]
        M.append(p)
    return keep


def _plan(train_rows, inputs):
    """Decide how each input column is used, from training rows.

    Returns (num, cat):
      num  maps each kept numeric column to (scale low, scale high, median).
      cat  maps each kept text column to its sorted training levels.
    """
    n = len(train_rows)
    num, cat = {}, {}
    for c in inputs:
        vals = [_finite(r.get(c)) for r in train_rows]
        present = [x for x in vals if x is not None]
        if present and len(present) >= NUMERIC_FRAC * n:
            if n >= 20 and len(set(present)) > ID_FRAC * n:
                continue  # identifier-like numeric column
            srt = sorted(present)
            lo = _quantile(srt, CLIP_Q)
            hi = _quantile(srt, 1.0 - CLIP_Q)
            if hi <= lo:
                lo, hi = srt[0], srt[-1]
            if hi > lo:
                num[c] = (lo, hi, srt[len(srt) // 2])
        else:
            levels = sorted({_text(r.get(c)) for r in train_rows})
            if 0 < len(levels) <= MAX_LEVELS:
                cat[c] = levels
    return num, cat


def _vector(r, num, cat):
    """Encoded inputs of one row. Numeric values are scaled by the training range
    (not clipped, so out-of-range values stay far from the training rows); missing
    numbers take the training median; text takes one-hot form, with unseen values
    giving an all-zero block."""
    vec = []
    for c, (lo, hi, med) in num.items():
        x = _finite(r.get(c))
        if x is None:
            x = med
        vec.append((x - lo) / (hi - lo))
    for c, levels in cat.items():
        s = _text(r.get(c))
        vec.extend(1.0 if s == lv else 0.0 for lv in levels)
    return vec


def _nearest(x, pool, skip=-1):
    """Smallest Euclidean distance from x to a pool point, leaving out index skip."""
    return min((math.dist(x, y) for j, y in enumerate(pool) if j != skip),
               default=math.inf)


def drift(train_csv, rows) -> int | None:
    """Index into `rows` of the first row at which the rows stop looking like
    train_csv, or None if they never do."""
    train_rows, inputs, goals = _load(train_csv)
    rows = list(rows)
    m = len(rows)
    n = len(train_rows)
    if m == 0 or n == 0:
        return None

    # Input novelty: a query is novel if its nearest pool row is farther away than
    # NOVEL_Q of the pool rows are from their own nearest pool neighbour.
    num, cat = _plan(train_rows, inputs)
    pool = list(range(n))
    if n > MAX_POOL:
        pool = sorted(random.Random(SEED).sample(pool, MAX_POOL))
    Xp = [_vector(train_rows[k], num, cat) for k in pool]
    has_x = bool(Xp) and len(Xp[0]) > 0

    novel = [False] * m
    if has_x and len(Xp) >= 2:
        Q = [_vector(r, num, cat) for r in rows]
        step = max(1, len(Xp) // CALIB)
        loo = sorted(_nearest(Xp[k], Xp, skip=k) for k in range(0, len(Xp), step))
        thresh = loo[min(len(loo) - 1, int(NOVEL_Q * (len(loo) - 1)))]
        novel = [_nearest(q, Xp) > thresh for q in Q]

    # Goal statistics from training (finite values only).
    names = [c for c, _ in goals]
    stats = {}
    for c in names:
        ys = [y for y in (_finite(r.get(c)) for r in train_rows) if y is not None]
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
            y = _finite(r.get(c))
            if y is not None:
                vals[c] = y
        qvals.append(vals)

    # Windows of WINDOW rows; if there are fewer rows, the whole list is one window.
    w = min(WINDOW, m)
    for i in range(w - 1, m):
        win = range(i - w + 1, i + 1)

        if sum(1 for k in win if novel[k]) >= NOVEL_FRAC * w:
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

