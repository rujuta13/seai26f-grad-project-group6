import csv

import numpy as np
from sklearn.cluster import KMeans


N_FAMILIES = 5
BEST_FRACTION = 0.05


def _to_value(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [h for h in (reader.fieldnames or [])]
        rows = [{k: _to_value(v) for k, v in r.items()} for r in reader]
    return header, rows


def _column_kind(name):
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


def _numeric_column(values):
    """Return a float array; non-numeric columns are category-coded."""
    if all(isinstance(v, float) for v in values):
        return np.asarray(values, dtype=float)
    labels = {}
    codes = []
    for v in values:
        key = str(v)
        if key not in labels:
            labels[key] = float(len(labels))
        codes.append(labels[key])
    return np.asarray(codes, dtype=float)


def _input_matrix(header, rows):
    cols = [c for c in header if _column_kind(c) == "input"]
    if not rows or not cols:
        return np.zeros((len(rows), 0))
    mats = [_numeric_column([r.get(c, 0.0) for r in rows]) for c in cols]
    return np.column_stack(mats)


def _goal_matrix(header, rows):
    """Return an (n, m) matrix where smaller is always better."""
    cols = [c for c in header if _column_kind(c) in ("max", "min")]
    if not rows or not cols:
        return np.zeros((len(rows), 0))
    mats = []
    for c in cols:
        vals = np.asarray([float(r[c]) for r in rows], dtype=float)
        if _column_kind(c) == "max":
            vals = -vals
        mats.append(vals)
    return np.column_stack(mats)


def families(train_csv) -> list[int]:
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
    header, rows = _load(train_csv)
    n = len(rows)
    cap = int(BEST_FRACTION * n)
    if n == 0 or cap == 0:
        return []
    G = _goal_matrix(header, rows)
    if G.shape[1] == 0:
        return []

    # Pareto front (non-dominated rows, minimisation on all goals).
    le = (G[:, None, :] <= G[None, :, :]).all(axis=-1)  # le[j, i]: G[j] <= G[i]
    lt = (G[:, None, :] < G[None, :, :]).any(axis=-1)   # lt[j, i]: G[j] <  G[i] somewhere
    dominated = (le & lt).any(axis=0)
    front = np.where(~dominated)[0]

    if len(front) > cap:
        # Rank each goal, sum ranks, keep the best overall within the cap.
        ranks = G.argsort(axis=0).argsort(axis=0)
        score = ranks.sum(axis=1)
        order = sorted(front.tolist(), key=lambda i: (score[i], i))
        front = np.asarray(order[:cap], dtype=int)

    return sorted(int(i) for i in front)

