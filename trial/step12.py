"""Step 12: guess. Estimate each goal of one player from his other numbers, with a
range that says how sure the guess is.

Method: ridge regression on the input columns of train_csv.
  - Numeric input columns are standardised; a missing cell is filled with the
    training median. Text input columns with a few levels are one-hot encoded.
    Text columns with many levels (names, urls, teams, nations) and
    identifier-like numeric columns (ids, ranks) are skipped, since a new
    player cannot supply them and they carry no player type.
  - Estimate: the ridge prediction for the row.
  - Range: the 5th..95th percentile of out-of-fold residuals from 5-fold
    cross-validation, so the range reflects the misses the model actually makes
    on players it has not seen. The range is widened when the row lacks many
    inputs, and clamped to the goal values seen in training.
  - If a goal has no usable training values, the range is (-inf, inf).
"""
import csv
import math
import os
import random

SEED = 1          # fixed seed so results are reproducible
MAX_FIT = 1000    # training rows used for fitting (a seeded sample if there are more)
FOLDS = 5         # cross-validation folds for the residuals
RIDGE = 1.0       # ridge penalty on the non-intercept weights
MAX_LEVELS = 20   # text columns with more levels are skipped (identifier-like)
ID_FRAC = 0.9     # numeric column with more distinct values than this share of rows is an id
NUMERIC_FRAC = 0.9  # a column is numeric if this share of its cells parse as numbers
Q_LO = 0.05       # lower residual quantile: the range covers 90% of out-of-fold misses
Q_HI = 0.95       # upper residual quantile
MISS_WIDEN = 2.0  # range widening per unit share of inputs the row lacks
_CACHE = {}


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _load(path):
    """Read a MOOT csv. Returns (rows, input_columns, goal_columns).
    goal_columns is a list of (name, sign): +1 maximise, -1 minimise."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    inputs, goals = [], []
    for c in cols:
        if c.endswith("+"):
            goals.append((c, 1))
        elif c.endswith("-"):
            goals.append((c, -1))
        elif c.endswith("X"):
            continue
        else:
            inputs.append(c)
    return rows, inputs, goals


def _quantile(vals, q):
    """Linear-interpolated quantile of a non-empty ascending list."""
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _spec(rows, inputs):
    """Decide which input columns to use and how to encode them, from training rows.
    Returns (numeric specs, categorical specs)."""
    n = len(rows)
    nums, cats = [], []
    for c in inputs:
        raw = [r.get(c) for r in rows]
        vals = [_finite(v) for v in raw]
        present = [x for x in vals if x is not None]
        if not present:
            continue
        if len(present) >= NUMERIC_FRAC * n:
            if n >= 20 and len(set(present)) > ID_FRAC * n:
                continue  # identifier-like numeric column
            median = sorted(present)[len(present) // 2]
            filled = [x if x is not None else median for x in vals]
            mean = sum(filled) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in filled) / n)
            if sd > 0:
                nums.append((c, mean, sd, median))
        else:
            levels = sorted({_text(v) for v in raw})
            if 2 <= len(levels) <= MAX_LEVELS:
                cats.append((c, levels))
    return nums, cats


def _design(r, nums, cats):
    """Feature vector for one row (intercept first), and the share of inputs it lacks."""
    x = [1.0]
    miss = 0
    total = 0
    for c, mean, sd, median in nums:
        total += 1
        v = _finite(r.get(c))
        if v is None:
            v = median
            miss += 1
        x.append((v - mean) / sd)
    for c, levels in cats:
        total += 1
        s = _text(r.get(c))
        if s == "" or s not in levels:
            miss += 1
        if s in levels:
            x.extend(1.0 if s == lv else 0.0 for lv in levels)
        else:
            x.extend(0.0 for _ in levels)
    share = miss / total if total else 0.0
    return x, share


def _solve(G, b):
    """Ridge solve (G + RIDGE * I, intercept unpenalised) w = b.
    G is stored upper-triangular (only entries with column >= row are read)."""
    d = len(b)
    A = [[0.0] * d for _ in range(d)]
    for a in range(d):
        for c in range(a, d):
            A[a][c] = G[a][c]
            A[c][a] = G[a][c]
        if a > 0:
            A[a][a] += RIDGE
    # Cholesky factorisation A = L L^T (A is positive definite thanks to RIDGE).
    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][t] * L[j][t] for t in range(j))
            if i == j:
                L[i][i] = math.sqrt(s) if s > 1e-12 else 1e-6
            else:
                L[i][j] = s / L[j][j]
    z = [0.0] * d
    for i in range(d):
        z[i] = (b[i] - sum(L[i][t] * z[t] for t in range(i))) / L[i][i]
    w = [0.0] * d
    for i in reversed(range(d)):
        w[i] = (z[i] - sum(L[t][i] * w[t] for t in range(i + 1, d))) / L[i][i]
    return w


def _fit(xs, ys, d):
    """Ridge fit on all rows, plus sorted out-of-fold residuals from k-fold CV."""
    m = len(ys)
    k = max(1, min(FOLDS, m))
    gram = [[[0.0] * d for _ in range(d)] for _ in range(k)]
    rhs = [[0.0] * d for _ in range(k)]
    for j in range(m):
        f = j % k
        x, y = xs[j], ys[j]
        g, b = gram[f], rhs[f]
        for a in range(d):
            xa = x[a]
            b[a] += xa * y
            if xa:
                row = g[a]
                for c in range(a, d):
                    row[c] += xa * x[c]

    G = [[0.0] * d for _ in range(d)]
    B = [0.0] * d
    for f in range(k):
        for a in range(d):
            B[a] += rhs[f][a]
            for c in range(a, d):
                G[a][c] += gram[f][a][c]
    w_all = _solve(G, B)

    res = []
    for f in range(k):
        Gf = [[G[a][c] - gram[f][a][c] for c in range(d)] for a in range(d)]
        Bf = [B[a] - rhs[f][a] for a in range(d)]
        w = _solve(Gf, Bf)
        for j in range(f, m, k):
            pred = sum(a * b for a, b in zip(w, xs[j]))
            res.append(ys[j] - pred)
    res.sort()
    return w_all, res


def _model(train_csv):
    """Cached fitted model for a training file."""
    stamp = (train_csv, os.path.getmtime(train_csv))
    hit = _CACHE.get(stamp)
    if hit is not None:
        return hit

    rows, inputs, goals = _load(train_csv)
    n = len(rows)
    names = [c for c, _ in goals]
    nums, cats = _spec(rows, inputs) if n else ([], [])
    fits = {name: None for name in names}

    if n and names:
        pool = list(range(n))
        if n > MAX_FIT:
            pool = sorted(random.Random(SEED).sample(pool, MAX_FIT))
        feats = {i: _design(rows[i], nums, cats)[0] for i in pool}
        d = 1 + len(nums) + sum(len(lv) for _, lv in cats)
        for name in names:
            xs, ys = [], []
            for i in pool:
                y = _finite(rows[i].get(name))
                if y is not None:
                    xs.append(feats[i])
                    ys.append(y)
            if not ys:
                continue
            all_y = [y for y in (_finite(r.get(name)) for r in rows) if y is not None]
            w, res = _fit(xs, ys, d)
            fits[name] = (w, res, min(all_y), max(all_y))

    model = {"nums": nums, "cats": cats, "names": names, "fits": fits}
    _CACHE.clear()
    _CACHE[stamp] = model
    return model


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for one player's inputs (goal columns already removed).

    Returns {goal: (estimate, low, high)} with low <= estimate <= high. The range
    is the spread of cross-validated misses on training players, widened when the
    row lacks many inputs. A goal with no usable training values gets (0.0, -inf, inf).
    """
    model = _model(train_csv)
    names = model["names"]
    if not names:
        return {}

    x, share = _design(row, model["nums"], model["cats"])
    widen = 1.0 + MISS_WIDEN * share
    out = {}
    for name in names:
        fit = model["fits"][name]
        if fit is None:
            out[name] = (0.0, -math.inf, math.inf)
            continue
        w, res, ylo, yhi = fit
        est = sum(a * b for a, b in zip(w, x))
        low = est + _quantile(res, Q_LO) * widen
        high = est + _quantile(res, Q_HI) * widen

        # Keep everything within the goal values seen in training, and ordered.
        est = min(max(est, ylo), yhi)
        low = min(max(low, ylo), yhi)
        high = min(max(high, ylo), yhi)
        low = min(low, est)
        high = max(high, est)
        out[name] = (float(est), float(low), float(high))
    return out

