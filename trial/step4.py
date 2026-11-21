import csv
import math
from collections import deque

import numpy as np


_ALPHA = 1e-4          # per-check significance for the drift warning
_WINDOW = 20           # number of most recent rows examined for drift
_BASE_Q = 0.95         # baseline quantile of nearest-neighbour distances


def _to_value(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [{k: _to_value(v) for k, v in r.items()} for r in reader]
    return header, rows


def _kind(name):
    if not name:
        return "ignore"
    last = name[-1]
    if last == "X":
        return "ignore"
    if last == "+":
        return "max"
    if last == "-":
        return "min"
    return "input"


def _as_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.inf
    return f if math.isfinite(f) else math.inf


# ---------------------------------------------------------------- stream

def _goal_setup(header):
    goals = [c for c in header if _kind(c) in ("max", "min")]
    sign = np.array(
        [-1.0 if _kind(c) == "max" else 1.0 for c in goals], dtype=float
    )
    return goals, sign


def _goal_vec(row, goals, sign):
    vals = np.array([_as_float(row.get(c)) for c in goals], dtype=float)
    # Negating an infinite value gives -inf for "max" goals with inf missing;
    # map missing/non-finite values to worst-possible instead.
    vals = np.where(np.isfinite(vals), vals * sign, np.inf)
    return vals


def stream(train_csv, rows) -> list[bool]:
    """Keep a row iff no earlier reference row (train or previously streamed)
    Pareto-dominates it on the goals. Only uses train_csv and earlier rows."""
    header, train = _load(train_csv)
    goals, sign = _goal_setup(header)
    decisions: list[bool] = []
    if not goals:
        return [True for _ in rows]

    ref_list = [_goal_vec(r, goals, sign) for r in train]
    ref = (
        np.array(ref_list, dtype=float).reshape(len(ref_list), len(goals))
        if ref_list
        else np.zeros((0, len(goals)), dtype=float)
    )

    for r in rows:
        g = _goal_vec(r, goals, sign)
        if ref.shape[0] == 0:
            keep = True
        else:
            le = (ref <= g).all(axis=1)   # ref weakly better everywhere
            lt = (ref < g).any(axis=1)    # ref strictly better somewhere
            keep = not bool((le & lt).any())
        decisions.append(bool(keep))
        # Domination is transitive, so adding every row is equivalent to
        # tracking only the Pareto front.
        ref = np.vstack([ref, g[None, :]])
    return decisions


# ----------------------------------------------------------------- drift

def _input_specs(header, train):
    """Per input column: (name, labels or None, fill value)."""
    specs = []
    for c in header:
        if _kind(c) != "input":
            continue
        values = [r.get(c) for r in train]
        if values and all(isinstance(v, float) for v in values):
            finite = [v for v in values if math.isfinite(v)]
            fill = float(np.mean(finite)) if finite else 0.0
            specs.append((c, None, fill))
        else:
            labels = {}
            for v in values:
                key = str(v)
                if key not in labels:
                    labels[key] = float(len(labels))
            specs.append((c, labels, 0.0))
    return specs


def _encode(row, specs):
    out = np.empty(len(specs), dtype=float)
    for j, (c, labels, fill) in enumerate(specs):
        v = row.get(c)
        if labels is None:
            f = v if isinstance(v, float) and math.isfinite(v) else fill
            out[j] = f
        else:
            out[j] = labels.get(str(v), float(len(labels)))
    return out


def _binom_tail(k, n, p):
    return sum(
        math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1)
    )


def _alarm_count(window, alpha):
    """Smallest k such that P(Binomial(window, 0.05) >= k) < alpha."""
    for k in range(1, window + 1):
        if _binom_tail(k, window, 1.0 - _BASE_Q) < alpha:
            return k
    return window + 1


def drift(train_csv, rows) -> int | None:
    """Warn at the first row index where the last _WINDOW rows are, as a group,
    far from the training inputs (nearest-neighbour distance above the
    training baseline more often than chance allows)."""
    header, train = _load(train_csv)
    specs = _input_specs(header, train)
    n = len(train)
    if n < 2 or not specs:
        return None

    X = np.vstack([_encode(r, specs) for r in train])
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd

    # Baseline: leave-one-out nearest-neighbour distance within train.
    diff = Z[:, None, :] - Z[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(D, np.inf)
    base = D.min(axis=1)
    threshold = float(np.quantile(base, _BASE_Q))

    k_alarm = _alarm_count(_WINDOW, _ALPHA)
    recent: deque[bool] = deque(maxlen=_WINDOW)

    for i, r in enumerate(rows):
        z = (_encode(r, specs) - mu) / sd
        d = float(np.sqrt(((Z - z) ** 2).sum(axis=1)).min())
        recent.append(d > threshold)
        if len(recent) == _WINDOW and sum(recent) >= k_alarm:
            return i
    return None

