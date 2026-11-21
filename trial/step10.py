"""Step 10: race our tuner and guesser against standard rivals on test_csv.

Tuner race: each method may measure at most BUDGET builds of test_csv and
returns the build it likes best. Its score is the regret: the normalised goal
sum of that build minus the best normalised goal sum in test_csv. Lower is
better, and 0 means the best build was found. Random search and local search
are averaged over SEEDS.

Guesser race: each method predicts every goal of every test build from its
inputs alone. Its score is the mean absolute error divided by each goal's
range over test_csv. Lower is better.

The standard library has no tuner or regression model, so the rivals are
stdlib-only stand-ins: a seeded local search over one-flag neighbours, and a
ridge regression on standardised numeric inputs and one-hot categorical inputs,
solved with Gaussian elimination.

Returned keys: "ours", "random_search", "local_search" (tuner regret),
"guesser_ours_nmae", "guesser_ridge_nmae" (guesser error). All values are
lower-is-better.
"""

import csv
import math
import os
import random
import tempfile

from step1 import _load, _split_columns
from step2 import guess
from step7 import choose

BUDGET = 30
SEEDS = range(10)
RIDGE_LAMBDA = 1.0

_KEYS = (
    "ours",
    "random_search",
    "local_search",
    "guesser_ours_nmae",
    "guesser_ridge_nmae",
)


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _normalised_scores(rows, maximise, minimise):
    """Per row: normalised goal sum in [0, 1] (lower is better), or None if a goal is missing."""
    names = maximise + minimise
    sign = [-1.0] * len(maximise) + [1.0] * len(minimise)
    raw = []
    for r in rows:
        vec = []
        for g, s in zip(names, sign):
            v = _finite(r.get(g))
            if v is None:
                vec = None
                break
            vec.append(s * v)
        raw.append(vec)

    usable = [v for v in raw if v is not None]
    if not usable:
        return [None] * len(rows)

    m = len(names)
    lo = [min(v[j] for v in usable) for j in range(m)]
    hi = [max(v[j] for v in usable) for j in range(m)]
    out = []
    for vec in raw:
        if vec is None:
            out.append(None)
            continue
        total = 0.0
        for j in range(m):
            if hi[j] > lo[j]:
                total += (vec[j] - lo[j]) / (hi[j] - lo[j])
        out.append(total / m)
    return out


def _write_blind(columns, rows, goals, path):
    """Write rows to path with every goal cell replaced by '?'."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {c: ("?" if c in goals else r.get(c, "")) for c in columns}
            )


def _random_search(valid, budget, rng, score):
    picks = rng.sample(valid, min(budget, len(valid)))
    return min(picks, key=lambda i: (score[i], i))


def _hamming(a, b, inputs):
    return sum(1 for c in inputs if str(a.get(c)) != str(b.get(c)))


def _local_search(rows, inputs, valid, budget, rng, score):
    """Start at a random build, then repeatedly measure an unmeasured one-flag
    neighbour of the current best (or any unmeasured build if none is left)."""
    unmeasured = set(valid)
    best = rng.choice(sorted(valid))
    unmeasured.discard(best)
    used = 1
    limit = min(budget, len(valid))
    while used < limit:
        near = sorted(
            i for i in unmeasured if _hamming(rows[i], rows[best], inputs) == 1
        )
        pool = near if near else sorted(unmeasured)
        pick = rng.choice(pool)
        unmeasured.discard(pick)
        used += 1
        if (score[pick], pick) < (score[best], best):
            best = pick
    return best


def _encode_fit(train, inputs):
    """Feature map fitted on train: numeric columns standardised, others one-hot."""
    spec = []
    for c in inputs:
        vals = [r.get(c) for r in train]
        if vals and all(isinstance(v, float) for v in vals):
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            spec.append((c, "num", (mean, sd if sd > 0 else 1.0)))
        else:
            levels = sorted({str(v) for v in vals})
            spec.append((c, "cat", levels))
    return spec


def _features(row, spec):
    feats = [1.0]
    for c, kind, data in spec:
        if kind == "num":
            mean, sd = data
            v = _finite(row.get(c))
            feats.append(0.0 if v is None else (v - mean) / sd)
        else:
            s = str(row.get(c))
            feats.extend(1.0 if s == lev else 0.0 for lev in data)
    return feats


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
        M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0 for i in range(n)
    ]


def _fit_ridge(X, y, lam):
    """Ridge weights; the bias (index 0) is not penalised."""
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for x, t in zip(X, y):
        for i in range(d):
            xi = x[i]
            b[i] += xi * t
            row = A[i]
            for j in range(d):
                row[j] += xi * x[j]
    for i in range(1, d):
        A[i][i] += lam
    return _solve(A, b)


def _fit_all(train, inputs, names):
    spec = _encode_fit(train, inputs)
    models = {}
    for g in names:
        X, y = [], []
        for r in train:
            t = _finite(r.get(g))
            if t is None:
                continue
            X.append(_features(r, spec))
            y.append(t)
        if X:
            models[g] = _fit_ridge(X, y, RIDGE_LAMBDA)
    return spec, models


def race(train_csv, test_csv) -> dict[str, float]:
    """Return {method: score} on test_csv. Every score is lower-is-better.

    "ours" is step7's choose(); "random_search" and "local_search" are the
    tuner rivals; "guesser_ours_nmae" is step2's guess(); "guesser_ridge_nmae"
    is a stdlib ridge regression. Guesser scores are mean absolute error
    divided by each goal's range over test_csv.
    """
    columns, test = _load(test_csv)
    maximise, minimise, inputs = _split_columns(columns)
    names = maximise + minimise
    goals = set(names)

    scores = _normalised_scores(test, maximise, minimise)
    valid = [i for i, s in enumerate(scores) if s is not None]
    if not valid or not names:
        # Nothing to compare: no test build has complete goal values.
        return {k: 0.0 for k in _KEYS}

    best_score = min(scores[i] for i in valid)

    def regret(j):
        if j is None or scores[j] is None:
            return 1.0
        return scores[j] - best_score

    result = {}

    # Tuner race. choose() needs a blind file with goals hidden.
    blind_fd, blind_path = tempfile.mkstemp(suffix=".csv")
    os.close(blind_fd)
    try:
        _write_blind(columns, test, goals, blind_path)

        def label(i):
            return {g: test[i].get(g) for g in names}

        ours_idx = choose(blind_path, label, budget=BUDGET)
        result["ours"] = regret(ours_idx)

        rand_regrets, local_regrets = [], []
        for seed in SEEDS:
            rng = random.Random(seed)
            rand_regrets.append(
                regret(_random_search(valid, BUDGET, rng, scores))
            )
            rng = random.Random(seed)
            local_regrets.append(
                regret(_local_search(test, inputs, valid, BUDGET, rng, scores))
            )
        result["random_search"] = sum(rand_regrets) / len(rand_regrets)
        result["local_search"] = sum(local_regrets) / len(local_regrets)
    finally:
        os.unlink(blind_path)

    # Guesser race.
    train_columns, train = _load(train_csv)
    _, _, train_inputs = _split_columns(train_columns)
    spec, models = _fit_all(train, train_inputs, names)

    spans = {}
    for g in names:
        vals = [v for v in (_finite(r.get(g)) for r in test) if v is not None]
        spans[g] = (max(vals) - min(vals)) if vals else 0.0

    ours_sum = 0.0
    ridge_sum = 0.0
    count = 0
    for r in test:
        probe = {c: r.get(c) for c in train_inputs}
        est = guess(train_csv, probe)
        feats = _features(probe, spec)
        for g in names:
            truth = _finite(r.get(g))
            if truth is None or g not in models or g not in est:
                continue
            scale = spans[g] if spans[g] > 0 else 1.0
            ours_sum += abs(est[g][0] - truth) / scale
            ridge_pred = sum(w * f for w, f in zip(models[g], feats))
            ridge_sum += abs(ridge_pred - truth) / scale
            count += 1

    result["guesser_ours_nmae"] = ours_sum / count if count else 0.0
    result["guesser_ridge_nmae"] = ridge_sum / count if count else 0.0
    return result

