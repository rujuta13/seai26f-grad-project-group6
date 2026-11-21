import csv
import math

import numpy as np


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _split_columns(cols):
    goal_max = [c for c in cols if c.endswith("+")]
    goal_min = [c for c in cols if c.endswith("-")]
    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]
    return inputs, goal_max, goal_min


def _goal_vector(row, goal_min, goal_max):
    """All goals as minimisation values; missing or non-finite = worst (+inf)."""
    vals = []
    for c in goal_min:
        v = _to_float(row.get(c))
        vals.append(v if math.isfinite(v) else math.inf)
    for c in goal_max:
        v = _to_float(row.get(c))
        vals.append(-v if math.isfinite(v) else math.inf)
    return vals


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) a row if no earlier reference row (training rows plus the
    rows before it in `rows`) strictly dominates it on all goals, i.e. it is
    on the running Pareto front. Only train_csv and prior rows are used."""
    cols, train_rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    n_goals = len(goal_min) + len(goal_max)

    if n_goals == 0:
        return [False] * len(rows)

    pool_list = [_goal_vector(r, goal_min, goal_max) for r in train_rows]
    pool = np.array(pool_list, dtype=float).reshape(len(pool_list), n_goals)

    out: list[bool] = []
    for r in rows:
        v = np.array(_goal_vector(r, goal_min, goal_max), dtype=float)
        if pool.shape[0] > 0:
            weakly = np.all(pool <= v, axis=1)
            strictly = np.any(pool < v, axis=1)
            dominated = bool(np.any(weakly & strictly))
        else:
            dominated = False
        out.append(not dominated)
        pool = np.vstack([pool, v[None, :]])
    return out


def _binom_tail(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    total = 0.0
    for j in range(k, n + 1):
        total += math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j))
    return total


def drift(train_csv, rows) -> int | None:
    """Index of the first row at which the last WINDOW rows contain too many
    'unusual' rows, relative to train_csv, or None if this never happens.

    Two signals, each tested with a one-sided binomial test:
      * input novelty: nearest-training-row distance above the 95th percentile
        of leave-one-out training distances;
      * goal outlier: some goal outside the train 2.5%-97.5% range.
    """
    cols, train_rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max

    if not train_rows or not rows:
        return None

    WINDOW = 30
    ALPHA = 5e-4
    P_IN = 0.05

    n_tr = len(train_rows)
    n_new = len(rows)

    # ---- input novelty flags ----
    flags_in = np.zeros(n_new, dtype=bool)
    d = len(inputs)
    if d > 0 and n_tr >= 2:
        X = np.array(
            [[_to_float(r.get(c)) for c in inputs] for r in train_rows],
            dtype=float,
        ).reshape(n_tr, d)
        X = np.where(np.isnan(X), 0.0, X)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Z = X / sd

        sq = (Z ** 2).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        D = np.maximum(D, 0.0)
        np.fill_diagonal(D, np.inf)
        loo = np.sqrt(D.min(axis=1))
        thr_in = float(np.quantile(loo, 0.95))

        Xn = np.array(
            [[_to_float(r.get(c)) for c in inputs] for r in rows],
            dtype=float,
        ).reshape(n_new, d)
        Xn = np.where(np.isnan(Xn), 0.0, Xn)
        Zn = Xn / sd
        for i in range(n_new):
            diff = Z - Zn[i]
            dist = math.sqrt(float((diff ** 2).sum(axis=1).min()))
            flags_in[i] = dist > thr_in

    # ---- goal outlier flags ----
    def _goal_flags(data_rows, lo_hi):
        flags = np.zeros(len(data_rows), dtype=bool)
        for i, r in enumerate(data_rows):
            for g, (lo, hi) in lo_hi.items():
                v = _to_float(r.get(g))
                if math.isfinite(v) and (v < lo or v > hi):
                    flags[i] = True
                    break
        return flags

    lo_hi = {}
    for g in goals:
        vals = np.array([_to_float(r.get(g)) for r in train_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            lo_hi[g] = (float(np.quantile(vals, 0.025)),
                        float(np.quantile(vals, 0.975)))

    if lo_hi:
        train_out_flags = _goal_flags(train_rows, lo_hi)
        p_out = float(train_out_flags.mean())
        p_out = min(max(p_out, 1e-3), 0.5)
        flags_out = _goal_flags(rows, lo_hi)
    else:
        p_out = 0.5
        flags_out = np.zeros(n_new, dtype=bool)

    # ---- sequential decision over sliding windows ----
    if n_new < WINDOW:
        return None

    for i in range(WINDOW - 1, n_new):
        lo_idx = i - WINDOW + 1
        k_in = int(flags_in[lo_idx:i + 1].sum())
        k_out = int(flags_out[lo_idx:i + 1].sum())
        if _binom_tail(WINDOW, k_in, P_IN) < ALPHA:
            return i
        if _binom_tail(WINDOW, k_out, p_out) < ALPHA:
            return i

    return None

