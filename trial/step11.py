import csv
import math

import numpy as np
from sklearn.cluster import KMeans


N_FAMILIES = 5
BEST_FRACTION = 0.05
_CHUNK = 1000        # rows per dominance-check block (bounds memory)
_ID_FRACTION = 0.9   # a column this unique is treated as an identifier
_MAX_LEVELS = 50     # categorical columns with more levels are skipped


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
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


def _num(cell):
    """Return cell as a finite float, or None if missing or not numeric."""
    if cell is None:
        return None
    try:
        f = float(cell)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _input_matrix(header, rows):
    """Standardised input matrix for clustering.

    Numeric inputs are kept (missing cells filled with the column mean).
    Categorical inputs with a modest number of levels are one-hot encoded.
    Identifier-like columns (nearly every value distinct, and for numbers
    all integers, e.g. row numbers, rank, names, urls) are dropped, since
    they carry no information about player type.
    """
    n = len(rows)
    blocks = []
    for c in header:
        if _kind(c) != "input":
            continue
        raw = [r.get(c, "") for r in rows]
        nums = [_num(v) for v in raw]
        present = [v for v in nums if v is not None]
        is_numeric = n > 0 and len(present) >= _ID_FRACTION * n
        distinct = len(set(str(v) for v in raw))

        if distinct > _ID_FRACTION * n and (
            not is_numeric or all(float(v).is_integer() for v in present)
        ):
            continue

        if is_numeric:
            fill = float(np.mean(present)) if present else 0.0
            col = np.asarray(
                [fill if v is None else v for v in nums], dtype=float
            )
            blocks.append(col[:, None])
        else:
            levels = sorted(set(str(v) for v in raw))
            if len(levels) > _MAX_LEVELS:
                continue
            index = {lab: j for j, lab in enumerate(levels)}
            onehot = np.zeros((n, len(levels)), dtype=float)
            for i, v in enumerate(raw):
                onehot[i, index[str(v)]] = 1.0
            blocks.append(onehot)

    if not blocks:
        return np.zeros((n, 0), dtype=float)

    X = np.hstack(blocks)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _goal_matrix(header, rows):
    """(n, m) matrix where smaller is always better; missing goals are worst."""
    goal_cols = [c for c in header if _kind(c) in ("max", "min")]
    n = len(rows)
    G = np.empty((n, len(goal_cols)), dtype=float)
    for j, c in enumerate(goal_cols):
        sign = -1.0 if _kind(c) == "max" else 1.0
        for i, r in enumerate(rows):
            v = _num(r.get(c))
            G[i, j] = math.inf if v is None else sign * v
    return G


def _pareto_front(G):
    """Indices of rows not dominated by any other row (minimisation)."""
    n = G.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for s in range(0, n, _CHUNK):
        Gi = G[s:s + _CHUNK]
        le = (G[None, :, :] <= Gi[:, None, :]).all(axis=2)  # G[j] <= G[i]
        lt = (G[None, :, :] < Gi[:, None, :]).any(axis=2)   # strictly better somewhere
        dominated[s:s + _CHUNK] = (le & lt).any(axis=1)
    return np.where(~dominated)[0]


def families(train_csv) -> list[int]:
    """One family label per data row of train_csv, in file order.

    Players are clustered on their standardised input attributes (goal
    columns are not used) with k-means into at most N_FAMILIES groups.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []
    X = _input_matrix(header, rows)
    if X.shape[1] == 0:
        return [0] * n
    k = min(N_FAMILIES, n)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    return [int(x) for x in labels]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows, at most 5% of rows.

    A row is good if no other row is at least as good on every goal and
    strictly better on one (Pareto front). If the front is larger than the
    cap, the rows with the best summed rank over the goals are kept.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    cap = int(BEST_FRACTION * n)
    goal_cols = [c for c in header if _kind(c) in ("max", "min")]
    if n == 0 or cap == 0 or not goal_cols:
        return []

    G = _goal_matrix(header, rows)
    front = _pareto_front(G)

    if len(front) > cap:
        ranks = G.argsort(axis=0, kind="stable").argsort(axis=0, kind="stable")
        score = ranks[front].sum(axis=1)
        order = sorted(zip(score.tolist(), front.tolist()))
        front = np.asarray([i for _, i in order[:cap]], dtype=int)

    return sorted(int(i) for i in front)

