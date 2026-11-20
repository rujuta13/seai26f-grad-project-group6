"""Step 10: race our tuner and guesser against standard-library rivals on test_csv.

race(train_csv, test_csv) returns {method: score}. Higher is better for every key.

Tuner race: the test builds form a blind pool, and each method may label at
most _BUDGET of them. The score is the chosen build's mean goal score over the
complete test builds, each goal scaled over those builds (0 worst, 1 best).
  "ours"                 step7.choose
  "random_search"        label random builds, keep the best labelled one
  "off_the_shelf_model"  refit a ridge regression after each label, label the
                         build it predicts best, keep the best labelled one

Guesser race: every test build's goal values are predicted from train_csv. The
score is minus the mean absolute error, each goal's error scaled by its
training range.
  "guesser_neg_mae"      step2.guess median estimate ("ours")
  "ridge_model_neg_mae"  ridge regression fitted on train_csv

The standard library has no hyper-parameter tuner or regression model, so the
rivals are small standard-library implementations written here.
"""
import csv
import math
import os
import random
import tempfile

from step1 import _load
from step2 import guess
from step7 import choose

_BUDGET = 30   # labels each tuner may spend on the test pool
_SEED = 1      # seed for the random baselines
_RIDGE = 1.0   # ridge penalty for the regression rival


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _level(v):
    """Comparable key for a cell: finite float, non-empty string, or None."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _goal_specs(header):
    """Return [(column, sign)]: +1.0 for '+' goals, -1.0 for '-' goals."""
    out = []
    for c in header:
        if c.endswith("+"):
            out.append((c, 1.0))
        elif c.endswith("-"):
            out.append((c, -1.0))
    return out


def _complete(rows, specs):
    """Indices of rows with every goal finite, and their oriented goal vectors."""
    keep, vecs = [], []
    for i, r in enumerate(rows):
        vec = []
        ok = True
        for c, sg in specs:
            f = _finite(r.get(c))
            if f is None:
                ok = False
                break
            vec.append(sg * f)
        if ok:
            keep.append(i)
            vecs.append(vec)
    return keep, vecs


def _combined(vecs):
    """Mean goal score of each oriented vector, each goal scaled over vecs (0 worst, 1 best)."""
    if not vecs:
        return []
    g = len(vecs[0])
    lo = [min(v[j] for v in vecs) for j in range(g)]
    hi = [max(v[j] for v in vecs) for j in range(g)]
    out = []
    for v in vecs:
        t = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            t += 1.0 if span == 0 else (v[j] - lo[j]) / span
        out.append(t / g)
    return out


def _best_of(ids, vecs):
    """Index (into vecs) of the best labelled build; ties go to the lower index."""
    comb = _combined([vecs[i] for i in ids])
    k = max(range(len(ids)), key=lambda t: (comb[t], -ids[t]))
    return ids[k]


def _write_blind(path, header, rows):
    """Write rows as a blind MOOT csv: goal cells become '?'."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            cells = []
            for c in header:
                if _is_goal(c):
                    cells.append("?")
                    continue
                v = r.get(c, "")
                cells.append(repr(v) if isinstance(v, float) else str(v))
            w.writerow(cells)


def _solve(A, b):
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if abs(p) < 1e-12:
            continue
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / p
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [
        M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0
        for i in range(n)
    ]


class _Encoder:
    """Turns input cells into numbers: numeric columns are standardised, others one-hot."""

    def __init__(self, rows, inputs):
        self.spec = []
        for c in inputs:
            cells = [_level(r.get(c)) for r in rows]
            cells = [v for v in cells if v is not None]
            nums = [v for v in cells if isinstance(v, float)]
            if cells and len(nums) == len(cells):
                mu = sum(nums) / len(nums)
                sd = math.sqrt(sum((v - mu) ** 2 for v in nums) / len(nums))
                self.spec.append(("num", c, (mu, sd if sd > 0.0 else 1.0)))
            else:
                levels = sorted(set(cells), key=lambda v: (isinstance(v, str), v))
                self.spec.append(("cat", c, {v: k for k, v in enumerate(levels)}))

    def encode(self, row):
        out = []
        for kind, c, data in self.spec:
            v = _level(row.get(c))
            if kind == "num":
                mu, sd = data
                out.append((v - mu) / sd if isinstance(v, float) else 0.0)
            else:
                cell = [0.0] * len(data)
                if v is not None and v in data:
                    cell[data[v]] = 1.0
                out.extend(cell)
        return out


class _Ridge:
    """Ridge regression with an unpenalised intercept, fitted by normal equations."""

    def __init__(self, X, y, lam=_RIDGE):
        n = len(X)
        d = len(X[0]) if X else 0
        self.ybar = sum(y) / n
        self.xbar = [sum(row[j] for row in X) / n for j in range(d)]
        A = [[0.0] * d for _ in range(d)]
        b = [0.0] * d
        for row, yv in zip(X, y):
            xc = [row[j] - self.xbar[j] for j in range(d)]
            yc = yv - self.ybar
            for j in range(d):
                xj = xc[j]
                if xj == 0.0:
                    continue
                b[j] += xj * yc
                for k in range(d):
                    A[j][k] += xj * xc[k]
        for j in range(d):
            A[j][j] += lam
        self.w = _solve(A, b)

    def predict(self, x):
        return self.ybar + sum(
            (x[j] - self.xbar[j]) * self.w[j] for j in range(len(self.w))
        )


def _random_search(n, vecs, budget):
    """Label budget random builds and return the best labelled one."""
    ids = random.Random(_SEED).sample(range(n), budget)
    return _best_of(ids, vecs)


def _model_search(X, vecs, budget):
    """Label a random third of the budget, then greedily label the builds the ridge model rates best."""
    n = len(X)
    labelled = random.Random(_SEED).sample(range(n), max(1, budget // 3))
    taken = set(labelled)
    while len(labelled) < budget and len(taken) < n:
        target = _combined([vecs[i] for i in labelled])
        model = _Ridge([X[i] for i in labelled], target)
        k = max(
            (j for j in range(n) if j not in taken),
            key=lambda j: (model.predict(X[j]), -j),
        )
        labelled.append(k)
        taken.add(k)
    return _best_of(labelled, vecs)


def _spans(rows, goal_names):
    """Training range of each goal (1.0 where the range is zero or unknown)."""
    out = []
    for c in goal_names:
        vals = []
        for r in rows:
            f = _finite(r.get(c))
            if f is not None:
                vals.append(f)
        span = (max(vals) - min(vals)) if vals else 0.0
        out.append(span if span > 0.0 else 1.0)
    return out


def _guesser_preds(train_csv, test, train_inputs, goal_names):
    """Per test row, the step2 median estimate of each goal (None if unavailable)."""
    preds = []
    for r in test:
        row = {c: r[c] for c in train_inputs if c in r}
        est = guess(train_csv, row)
        preds.append([est[c][0] if c in est else None for c in goal_names])
    return preds


def _ridge_preds(train, test, train_inputs, goal_names):
    """Per test row, a ridge prediction of each goal fitted on train (None if unavailable)."""
    enc = _Encoder(train, train_inputs)
    X_train = [enc.encode(r) for r in train]
    X_test = [enc.encode(r) for r in test]
    preds = [[None] * len(goal_names) for _ in test]
    for j, c in enumerate(goal_names):
        idx = [i for i, r in enumerate(train) if _finite(r.get(c)) is not None]
        if not idx:
            continue
        X = [X_train[i] for i in idx]
        y = [_finite(train[i][c]) for i in idx]
        model = _Ridge(X, y)
        for k, x in enumerate(X_test):
            preds[k][j] = model.predict(x)
    return preds


def _neg_mae(preds, test, goal_names, spans):
    """Minus the mean absolute error over all predicted goal cells, scaled by training range."""
    total = 0.0
    count = 0
    for pred, row in zip(preds, test):
        for j, c in enumerate(goal_names):
            t = _finite(row.get(c))
            if t is None or pred[j] is None:
                continue
            total += abs(pred[j] - t) / spans[j]
            count += 1
    if count == 0:
        return -math.inf
    return -total / count


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores (higher is better) for our tuner and guesser and the stdlib rivals.

    Keys: "ours", "random_search", "off_the_shelf_model" (tuner race on the
    test pool, score = mean goal score of the chosen build) and
    "guesser_neg_mae", "ridge_model_neg_mae" (prediction race, score = minus
    the mean scaled absolute error on the test goals).
    """
    header, train = _load(train_csv)
    theader, test = _load(test_csv)
    specs = _goal_specs(theader)
    train_inputs = [c for c in header if _is_input(c)]
    test_inputs = [c for c in theader if _is_input(c)]
    goal_names = [c for c, _ in _goal_specs(header)]

    scores = {
        "ours": 0.0,
        "random_search": 0.0,
        "off_the_shelf_model": 0.0,
        "guesser_neg_mae": -math.inf,
        "ridge_model_neg_mae": -math.inf,
    }

    # Tuner race on the complete test builds.
    keep, vecs = _complete(test, specs)
    pool = [test[i] for i in keep]
    n = len(pool)
    if n > 0 and specs:
        pool_score = _combined(vecs)
        budget = min(_BUDGET, n)
        goal_cols = [c for c, _ in specs]

        def label(i):
            return {c: pool[i][c] for c in goal_cols}

        with tempfile.TemporaryDirectory() as tmp:
            blind = os.path.join(tmp, "blind.csv")
            _write_blind(blind, theader, pool)
            ours = choose(blind, label, budget=budget)
        scores["ours"] = float(pool_score[ours])

        scores["random_search"] = float(
            pool_score[_random_search(n, vecs, budget)]
        )

        enc = _Encoder(pool, test_inputs)
        X = [enc.encode(r) for r in pool]
        scores["off_the_shelf_model"] = float(
            pool_score[_model_search(X, vecs, budget)]
        )

    # Guesser race: predict every test build's goals from the training builds.
    if goal_names:
        spans = _spans(train, goal_names)
        g_preds = _guesser_preds(train_csv, test, train_inputs, goal_names)
        scores["guesser_neg_mae"] = float(
            _neg_mae(g_preds, test, goal_names, spans)
        )
        r_preds = _ridge_preds(train, test, train_inputs, goal_names)
        scores["ridge_model_neg_mae"] = float(
            _neg_mae(r_preds, test, goal_names, spans)
        )

    return scores

