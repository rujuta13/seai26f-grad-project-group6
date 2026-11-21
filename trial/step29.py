"""Step 29: spot. Suspicion scores for customer records that look wrong (impossible
ages, odd incomes, typos) or that behave unlike any other customer.

odd(train_csv, rows) returns one non-negative score per row (higher = more
suspicious). Ordinary rows score near 0. The score adds up these checks:
  - Unusable cells: a goal or numeric input cell that is missing or not a finite
    number, and a text input value that never appears in train_csv.
  - Impossible values: an AGE below 0 or above MAX_AGE, and a negative value in a
    column that cannot be negative (INCOME, MNT spending, NUM counts, KIDHOME,
    TEENHOME, RECENCY, CUSTOMER_DAYS). These rules come from the column names.
  - Input range: a numeric input outside the training min..max, measured in
    training spans, so an extra digit lands far outside.
  - Robust outliers: a numeric input or goal far from the training median in MAD
    units. Skipped for columns with few distinct training values (flags, small
    counts), where the check would only flag legitimate rare values.
  - Input novelty: distance to the nearest training customer in min-max scaled
    input space, beyond the 95th percentile of training leave-one-out distances.
  - Goal disagreement: a goal far from the goals of the nearest training customers
    with similar inputs, in units of their spread.

Standard library only. Ordinary rows score near 0, and a fixed seed keeps the
sampled neighbour pool reproducible.
"""
import math
import random

from step1 import _load

SEED = 1              # fixed seed so the sampled neighbour pool is reproducible
MAX_POOL = 1000       # training rows used for neighbour search (seeded sample if more)
CALIB = 200           # pool rows used to calibrate input novelty
NOVEL_Q = 0.95        # quantile of leave-one-out distances defining "normal"
K_NEAR = 15           # nearest training customers used for the goal-disagreement check
NUMERIC_FRAC = 0.9    # a column is numeric if this share of its training cells parse
MAX_LEVELS = 25       # text columns with more levels are not used for unseen checks
MIN_DISTINCT = 10     # robust checks need at least this many distinct training values
ZOUT_OK = 6.0         # robust z-score treated as normal
RESID_OK = 3.0        # goal disagreement, in neighbour-spread units, treated as normal
CAP = 10.0            # cap on any single continuous term
MAX_AGE = 120.0       # oldest plausible customer age
NONNEG = ("INCOME", "MNT", "NUM", "KIDHOME", "TEENHOME", "RECENCY", "CUSTOMER_DAYS")
AGE = "AGE"
W_MISSING = 10.0      # missing or non-numeric goal cell
W_BAD_INPUT = 2.0     # missing or non-numeric numeric input cell
W_UNSEEN = 2.0        # text input value never seen in training
W_IMPOSSIBLE = 5.0    # value that cannot occur (negative amount, age outside 0..MAX_AGE)
W_RANGE = 1.0         # weight per training span a numeric input lies beyond the range
W_ROBUST = 1.0        # weight of a robust (MAD) outlier term
W_NOVEL = 1.0         # weight of the input-novelty term
W_DISAGREE = 1.0      # weight of the goal-disagreement term
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


def _base(name):
    """Column name without its goal sign (+ or -), in upper case."""
    return name.rstrip("+-").strip().upper()


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


def _limits(name):
    """(lowest, highest) plausible value implied by a column name; None means no limit."""
    b = _base(name)
    lo = 0.0 if b.startswith(NONNEG) else None
    hi = MAX_AGE if b.startswith(AGE) else None
    return lo, hi


def _numeric_present(raw):
    """Finite values of a column if the column is numeric (mostly parses), else None."""
    vals = [_finite(v) for v in raw]
    present = [x for x in vals if x is not None]
    if present and len(present) >= NUMERIC_FRAC * len(raw):
        return present
    return None


def _num_stats(present):
    """Training statistics for one numeric column."""
    srt = sorted(present)
    med = _quantile(srt, 0.5)
    mad = _quantile(sorted(abs(x - med) for x in srt), 0.5)
    rsd = 1.4826 * mad
    many = len(set(srt)) >= MIN_DISTINCT
    return {
        "lo": srt[0],
        "hi": srt[-1],
        "med": med,
        "rsd": rsd,
        "many": many,
        "robust": many and rsd > 0,
    }


def _plan(train_rows, inputs, goals):
    """Decide how each column is used, from training rows.

    num  maps numeric input columns to their training statistics.
    cat  maps text input columns (few levels) to their sorted training levels.
    gnum maps numeric goal columns to their training statistics.
    """
    num, cat, gnum = {}, {}, {}
    for c in inputs:
        raw = [r.get(c) for r in train_rows]
        present = _numeric_present(raw)
        if present is not None:
            num[c] = _num_stats(present)
        else:
            levels = sorted({_text(v) for v in raw})
            if len(levels) <= MAX_LEVELS:
                cat[c] = levels
    for c, _ in goals:
        present = _numeric_present([r.get(c) for r in train_rows])
        if present is not None:
            gnum[c] = _num_stats(present)
    return num, cat, gnum


def _cells(r, num, cat):
    """Encoded inputs of one row (the novelty space) and the penalty for its input cells.

    Numeric values are scaled by the training min..max without clipping, so values
    outside the training range stay far from the training customers. A missing
    numeric cell takes the training median in the encoding.
    """
    vec, pen = [], 0.0
    for c, st in num.items():
        x = _finite(r.get(c))
        span = st["hi"] - st["lo"]
        if x is None:
            pen += W_BAD_INPUT
            x = st["med"]
        else:
            lo_ok, hi_ok = _limits(c)
            if (lo_ok is not None and x < lo_ok) or (hi_ok is not None and x > hi_ok):
                pen += W_IMPOSSIBLE
            beyond = max(st["lo"] - x, x - st["hi"], 0.0)
            if beyond > 0:
                scale = span if span > 0 else max(abs(st["hi"]), 1.0)
                pen += W_RANGE * min(CAP, beyond / scale)
            if st["robust"]:
                dev = abs(x - st["med"]) / st["rsd"]
                pen += W_ROBUST * min(CAP, max(0.0, dev - ZOUT_OK))
        vec.append((x - st["lo"]) / span if span > 0 else 0.0)

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
    return min((math.dist(x, y) for j, y in enumerate(pool) if j != skip),
               default=math.inf)


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more
    likely a glitch, a typo, or a customer unlike anything in train_csv."""
    train_rows, inputs, goals = _load(train_csv)
    rows = list(rows)
    m = len(rows)
    scores = [0.0] * m
    n = len(train_rows)
    if m == 0 or n == 0:
        return scores

    num, cat, gnum = _plan(train_rows, inputs, goals)

    # Neighbour pool: all training rows, or a seeded sample of MAX_POOL of them.
    pool = list(range(n))
    if n > MAX_POOL:
        pool = sorted(random.Random(SEED).sample(pool, MAX_POOL))
    P = len(pool)
    Xp = [_cells(train_rows[k], num, cat)[0] for k in pool]
    has_x = P > 0 and len(Xp[0]) > 0
    Yp = {c: [_finite(train_rows[k].get(c)) for k in pool] for c in gnum}

    # Input novelty calibration: leave-one-out nearest distances within the pool.
    novel_on = has_x and P >= 2
    nscale = 1.0
    if novel_on:
        step = max(1, P // CALIB)
        loo = sorted(_nearest(Xp[k], Xp, skip=k) for k in range(0, P, step))
        thresh = _quantile(loo, NOVEL_Q)
        nscale = thresh if thresh > 0 else 1.0

    for i, r in enumerate(rows):
        vec, s = _cells(r, num, cat)

        dist = None
        if has_x:
            dist = [math.dist(vec, x) for x in Xp]
            if novel_on:
                nd = min(dist)
                s += W_NOVEL * min(CAP, max(0.0, nd / nscale - 1.0))

        order = None
        for c, _ in goals:
            st = gnum.get(c)
            if st is None:
                continue  # goal is not numeric in training, nothing to compare
            y = _finite(r.get(c))
            if y is None:
                s += W_MISSING
                continue

            lo_ok, _ = _limits(c)
            if lo_ok is not None and y < lo_ok:
                s += W_IMPOSSIBLE

            # Goal outlier against the training median, in robust units.
            if st["robust"]:
                dev = abs(y - st["med"]) / st["rsd"]
                s += W_ROBUST * min(CAP, max(0.0, dev - ZOUT_OK))

            # Goal disagreement with the goals of the nearest similar training customers.
            if dist is not None and st["many"]:
                if order is None:
                    order = sorted(range(P), key=dist.__getitem__)
                yk_list = Yp[c]
                nb = []
                for k in order:
                    yk = yk_list[k]
                    if yk is not None:
                        nb.append((dist[k], yk))
                        if len(nb) == K_NEAR:
                            break
                if nb:
                    est = _wmedian([(yk, 1.0 / (1.0 + d)) for d, yk in nb])
                    ys = sorted(yk for _, yk in nb)
                    iqr = _quantile(ys, 0.75) - _quantile(ys, 0.25)
                    spread = max(iqr, 0.25 * st["rsd"], 0.01 * abs(est), EPS)
                    s += W_DISAGREE * min(CAP, max(0.0, abs(y - est) / spread - RESID_OK))

        scores[i] = float(s)
    return scores

