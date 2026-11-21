"""Step 19: spot. Suspicion scores for player records that look like data-entry
mistakes, or that are so unusual their ratings should be checked twice.

odd(train_csv, rows) returns one non-negative score per row (higher = more
suspicious); ordinary rows score near 0. The score adds up these checks:
  - Unusable cells: a goal cell or a numeric input cell that is missing or not a
    finite number, and a text input value that never appears in train_csv.
  - Input range: a numeric input outside the training min..max, measured in
    training spans (catches a typo such as an extra digit in an attribute).
  - Input novelty: distance to the nearest training row in the encoded input
    space, beyond the 95th percentile of training leave-one-out distances.
  - Goal outlier: a goal far from the training median in robust (MAD) units.
  - Goal disagreement: a goal far from the goals of the nearest training rows
    with similar inputs, in units of their spread.

Encoding follows step11 and step12 so player files work. Identifier-like numeric
columns (ids, ranks) and text columns with many levels (names, urls, teams,
nations) are not used for distances or unseen checks, since a new player cannot
be judged on them. Numeric inputs are min-max scaled over train_csv.

For large training files, neighbours are searched in a seeded sample of at most
MAX_POOL training rows, which keeps the run time bounded. Missing or unusable
cells never crash the function.
"""
import math
import random

from step11 import _finite, _load

SEED = 1           # fixed seed so results are reproducible
NUMERIC_FRAC = 0.9  # a column is numeric if this share of its training cells parse as numbers
ID_FRAC = 0.9      # numeric column with more distinct values than this share of rows is an id
MAX_LEVELS = 25    # text columns with more levels are identifier-like and are not used
MAX_POOL = 1000    # training rows used for neighbour search (seeded sample if more)
CALIB = 200        # pool rows used to calibrate input novelty
NOVEL_Q = 0.95     # quantile of leave-one-out distances defining "normal"
K_NEAR = 15        # nearest training rows used for the goal-disagreement check
RESID_OK = 3.0     # goal disagreement, in neighbour-spread units, treated as normal
ZOUT_OK = 6.0      # robust z-score of a goal treated as normal
CAP_SPAN = 10.0    # cap on any single continuous term
W_MISSING = 10.0   # penalty for a missing or non-numeric goal cell
W_BAD_INPUT = 2.0  # penalty for a missing or non-numeric numeric input cell
W_UNSEEN = 2.0     # penalty per text input whose value never appears in training
W_RANGE = 1.0      # weight per training span a numeric input lies beyond the range
W_NOVEL = 1.0      # weight of the input-novelty term
EPS = 1e-12


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


def _wmedian(pairs):
    """Weighted median of (value, weight) pairs with positive weights."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def _cap(x):
    return min(CAP_SPAN, x)


def _plan(train_rows, inputs):
    """Decide how each input column is used, from training rows.

    Returns (num, cat):
      num  maps each kept numeric column to (min, max, median) over training rows.
      cat  maps each kept text column to its sorted training levels.
    Identifier-like numeric columns and many-level text columns are left out.
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
            num[c] = (srt[0], srt[-1], srt[len(srt) // 2])
        else:
            levels = sorted({_text(r.get(c)) for r in train_rows})
            if 0 < len(levels) <= MAX_LEVELS:
                cat[c] = levels
    return num, cat


def _vector(r, num, cat):
    """Encoded inputs of one row, and its cell-level penalty (missing or unusable
    numbers, out-of-range numbers, unseen text values)."""
    vec = []
    pen = 0.0
    for c, (lo, hi, med) in num.items():
        span = hi - lo
        x = _finite(r.get(c))
        if x is None:
            pen += W_BAD_INPUT
            x = med
        elif x < lo or x > hi:
            scale = span if span > 0 else max(abs(lo), 1.0)
            beyond = (lo - x) if x < lo else (x - hi)
            pen += W_RANGE * min(CAP_SPAN, beyond / scale)
        vec.append((x - lo) / span if span > 0 else 0.0)
    for c, levels in cat.items():
        s = _text(r.get(c))
        if s in levels:
            vec.extend(1.0 if s == lv else 0.0 for lv in levels)
        else:
            pen += W_UNSEEN
            vec.extend(0.0 for _ in levels)
    return vec, pen


def _nearest(x, pool, skip=-1):
    """Smallest Euclidean distance from x to a pool point, leaving out index skip."""
    best = math.inf
    for j, y in enumerate(pool):
        if j != skip:
            d = math.dist(x, y)
            if d < best:
                best = d
    return best


def _goal_stats(ys):
    """(median, robust spread) of a non-empty list of goal values, or None if empty.
    The spread is 1.4826 * MAD, falling back to the standard deviation if that is 0."""
    if not ys:
        return None
    srt = sorted(ys)
    med = _quantile(srt, 0.5)
    rsd = 1.4826 * _quantile(sorted(abs(y - med) for y in ys), 0.5)
    if rsd <= 0 and len(ys) > 1:
        mu = sum(ys) / len(ys)
        rsd = math.sqrt(sum((y - mu) ** 2 for y in ys) / (len(ys) - 1))
    return med, rsd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more
    likely a glitch, a typo, or unlike anything in train_csv."""
    rows = list(rows)
    m = len(rows)
    scores = [0.0] * m
    if m == 0:
        return scores

    train_rows, inputs, goals = _load(train_csv)
    n = len(train_rows)
    if n == 0:
        return scores

    num, cat = _plan(train_rows, inputs)

    # Neighbour pool: all training rows, or a seeded sample of MAX_POOL of them.
    pool = list(range(n))
    if n > MAX_POOL:
        pool = sorted(random.Random(SEED).sample(pool, MAX_POOL))
    Xp = [_vector(train_rows[k], num, cat)[0] for k in pool]
    has_x = bool(Xp) and len(Xp[0]) > 0
    Yp = [[_finite(train_rows[k].get(c)) for k in pool] for c, _ in goals]
    gstats = [_goal_stats([y for y in (_finite(r.get(c)) for r in train_rows)
                           if y is not None])
              for c, _ in goals]

    # Input novelty calibration: leave-one-out nearest distances within the pool.
    novel_on = has_x and len(Xp) >= 2
    nscale = 1.0
    if novel_on:
        step = max(1, len(Xp) // CALIB)
        loo = sorted(_nearest(Xp[k], Xp, skip=k) for k in range(0, len(Xp), step))
        thresh = _quantile(loo, NOVEL_Q)
        nscale = thresh if thresh > 0 else 1.0

    for i, r in enumerate(rows):
        vec, pen = _vector(r, num, cat)
        s = pen

        # Nearest pool rows in input space, sorted by distance.
        dl = None
        if has_x:
            dl = sorted((math.dist(vec, x), k) for k, x in enumerate(Xp))
            if novel_on:
                nd = dl[0][0]
                s += W_NOVEL * _cap(max(0.0, nd / nscale - 1.0))

        for j, (c, _) in enumerate(goals):
            y = _finite(r.get(c))
            if y is None:
                s += W_MISSING
                continue
            st = gstats[j]
            if st is None:
                continue  # no training values for this goal, nothing to compare
            med, rsd = st

            # Goal outlier against the training median, in robust units.
            dev = abs(y - med)
            if rsd > 0:
                s += _cap(max(0.0, dev / rsd - ZOUT_OK))
            elif dev > EPS * max(1.0, abs(med)):
                s += CAP_SPAN

            # Goal disagreement with the goals of the nearest similar training rows.
            if dl is not None:
                nb = []
                for d, k in dl:
                    yk = Yp[j][k]
                    if yk is not None:
                        nb.append((d, yk))
                        if len(nb) == K_NEAR:
                            break
                if nb:
                    est = _wmedian([(yk, 1.0 / (1.0 + d)) for d, yk in nb])
                    srt = sorted(yk for _, yk in nb)
                    iqr = _quantile(srt, 0.75) - _quantile(srt, 0.25)
                    spread = max(iqr, 0.25 * rsd, 0.01 * abs(est), EPS)
                    s += _cap(max(0.0, abs(y - est) / spread - RESID_OK))

        scores[i] = float(s)
    return scores

