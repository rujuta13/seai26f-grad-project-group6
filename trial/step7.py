import csv
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor


_KAPPA = 1.0        # exploration weight on tree-disagreement (lower score is better)
_N_TREES = 200
_SEED = 0


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


def _load_blind(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def _encode_inputs(header, rows):
    """Numeric matrix of input columns; non-numeric columns are category-coded."""
    cols = [c for c in header if _kind(c) == "input"]
    n = len(rows)
    if not cols or n == 0:
        return np.zeros((n, 1), dtype=float)

    mats = []
    for c in cols:
        raw = [r.get(c, "") for r in rows]
        try:
            vals = np.asarray([float(v) for v in raw], dtype=float)
            if not np.all(np.isfinite(vals)):
                raise ValueError
        except (TypeError, ValueError):
            labels = {}
            codes = []
            for v in raw:
                key = str(v)
                if key not in labels:
                    labels[key] = float(len(labels))
                codes.append(labels[key])
            vals = np.asarray(codes, dtype=float)
        mats.append(vals)

    X = np.column_stack(mats)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _farthest_point_init(X, k, start=0):
    """Pick k rows spread across the input space, deterministically."""
    n = X.shape[0]
    chosen = [start]
    mind = np.sqrt(((X - X[start]) ** 2).sum(axis=1))
    mind[start] = -1.0
    while len(chosen) < k:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        d = np.sqrt(((X - X[nxt]) ** 2).sum(axis=1))
        mind = np.minimum(mind, d)
        for c in chosen:
            mind[c] = -1.0
    return chosen


def _rank_score(M):
    """Mean normalised rank over goal columns (0 = best). M has shape (k, m)."""
    k = M.shape[0]
    if k <= 1:
        return np.zeros(k, dtype=float)
    ranks = M.argsort(axis=0, kind="stable").argsort(axis=0, kind="stable").astype(float)
    return ranks.mean(axis=1) / (k - 1)


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, labelling at most `budget` rows.

    blind_csv: CSV whose goal cells are all '?'.
    label(i):  returns a dict {goal column: value} for row i. Called at most
               `budget` times; a further call raises RuntimeError.
    goal:      optional name of one goal column to optimise; otherwise all
               goal columns are optimised together (mean normalised rank).
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer")

    header, rows = _load_blind(blind_csv)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no rows")

    all_goals = [c for c in header if _kind(c) in ("max", "min")]
    if goal is None:
        goal_cols = all_goals
    else:
        if goal not in all_goals:
            raise ValueError(f"'{goal}' is not a goal column in {blind_csv}")
        goal_cols = [goal]
    if not goal_cols:
        raise ValueError("no goal columns (names ending in '+' or '-') found")

    signs = np.array(
        [-1.0 if _kind(c) == "max" else 1.0 for c in goal_cols], dtype=float
    )
    X = _encode_inputs(header, rows)

    target = min(budget, n)
    labeled = {}   # row index -> signed goal vector (smaller is better)

    def do_label(i):
        if len(labeled) >= budget:
            raise RuntimeError("labelling budget exhausted")
        result = label(i)
        vec = np.empty(len(goal_cols), dtype=float)
        for j, c in enumerate(goal_cols):
            try:
                v = float(result.get(c))
                if not math.isfinite(v):
                    v = math.inf
            except (TypeError, ValueError, AttributeError):
                v = math.inf
            vec[j] = signs[j] * v
        labeled[i] = vec

    # Phase 1: spread-out initial sample.
    n_init = min(target, max(3, target // 4))
    for i in _farthest_point_init(X, n_init, start=0):
        do_label(int(i))

    # Phase 2: model-guided sequential labelling.
    while len(labeled) < target:
        L = sorted(labeled)
        U = [i for i in range(n) if i not in labeled]
        if not U:
            break

        M = np.vstack([labeled[i] for i in L])
        scores = _rank_score(M)

        if len(L) < 2:
            # Too little data to model; continue spreading out.
            d = np.sqrt(((X[U][:, None, :] - X[L][None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
            pick = U[int(np.argmax(d))]
        else:
            model = RandomForestRegressor(
                n_estimators=_N_TREES,
                min_samples_leaf=1,
                random_state=_SEED,
                n_jobs=1,
            ).fit(X[L], scores)
            tree_preds = np.stack([t.predict(X[U]) for t in model.estimators_])
            mean = tree_preds.mean(axis=0)
            spread = tree_preds.std(axis=0)
            acq = mean - _KAPPA * spread
            pick = U[int(np.argmin(acq))]

        do_label(int(pick))

    L = sorted(labeled)
    M = np.vstack([labeled[i] for i in L])
    final_scores = _rank_score(M)
    return int(L[int(np.argmin(final_scores))])

