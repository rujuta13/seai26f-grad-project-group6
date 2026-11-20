"""Step 10: race. Benchmark our tuner (step7.choose) against a random search and a
hand-written stdlib hill climb, and our guesser (step2.guess) against a stdlib
ridge regression, on test_csv.

The standard library ships no tuner and no regression model, so the two rivals
are stand-ins written here with only the standard library:
  - stdlib_hill_climb: greedy walk through the candidate pool. From the best build
    benchmarked so far, benchmark the nearest unbenchmarked build in raw input
    cells (fewest differing cells). This is a simple local-search tuner.
  - linear_predict_r2: ridge-regularised least squares on the one-hot encoded
    inputs (step1._encode), fitted on train_csv.

Tuning scores use the step1 scale on test_csv: each goal is min-max scaled over
the test builds with complete goal values (0 = worst, 1 = best) and averaged.
Each method sees only the builds it benchmarks. The returned score is the true
score of the build it picks. "oracle" is the best true score in test_csv, an
upper bound no method can exceed.

Prediction scores are R^2 per goal on test_csv (higher is better). "ours_predict_r2"
and "linear_predict_r2" are means over goals; per-goal keys carry a ":goal" suffix.

All returned values are higher-is-better.
"""
import csv
import math
import os
import random
import tempfile

from step1 import _load, _encode
from step2 import guess
from step7 import choose, _scores

BUDGET = 30    # benchmark builds allowed for each tuner
SEED = 1       # fixed seed so results are reproducible
RIDGE = 1e-6   # tiny ridge term keeps the least-squares system solvable


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _goal_values(row, goals):
    """Goal cells of a row as {name: float}, leaving out unusable cells."""
    out = {}
    for c, _ in goals:
        x = _finite(row.get(c))
        if x is not None:
            out[c] = x
    return out


def _oriented(row, goals):
    """Goal vector with higher always better, or None if any goal cell is unusable."""
    vals = []
    for c, sign in goals:
        x = _finite(row.get(c))
        if x is None:
            return None
        vals.append(sign * x)
    return vals


def _true_scores(rows, goals):
    """Step1-scale score of every row over all rows (None where unusable)."""
    ori = [_oriented(r, goals) for r in rows]
    usable = [o for o in ori if o is not None]
    if not usable:
        return [None] * len(rows)
    k = len(goals)
    lo = [min(o[j] for o in usable) for j in range(k)]
    hi = [max(o[j] for o in usable) for j in range(k)]
    out = []
    for o in ori:
        if o is None:
            out.append(None)
            continue
        total = 0.0
        for j in range(k):
            span = hi[j] - lo[j]
            if span > 0:
                total += (o[j] - lo[j]) / span
        out.append(total / k)
    return out


def _truth_at(truth, i):
    s = truth[i] if 0 <= i < len(truth) else None
    return 0.0 if s is None else s


def _pick(labelled, goals, fallback):
    """Best benchmarked index under step7's scale, mirroring choose()'s final pick."""
    scores = _scores(labelled, goals)
    if not scores:
        return fallback
    return max(scores, key=lambda i: (scores[i], -i))


def _random_search(n, label, goals, budget):
    rng = random.Random(SEED)
    picks = rng.sample(range(n), budget)
    labelled = {i: label(i) for i in picks}
    return _pick(labelled, goals, picks[0])


def _hill_climb(rows, inputs, label, goals, budget):
    """Stdlib stand-in tuner: walk to the nearest unbenchmarked build of the best so far."""
    n = len(rows)
    rng = random.Random(SEED)
    start = rng.randrange(n)
    labelled = {start: label(start)}
    while len(labelled) < budget:
        best_i = _pick(labelled, goals, start)
        base = rows[best_i]
        cand = [i for i in range(n) if i not in labelled]
        nxt = min(cand, key=lambda i: (sum(1 for c in inputs
                                           if rows[i].get(c) != base.get(c)), i))
        labelled[nxt] = label(nxt)
    return _pick(labelled, goals, start)


def _write_blind(test_csv, goal_names, path):
    """Copy test_csv with every goal cell set to '?', keeping the header and row order."""
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        with open(path, "w", newline="") as g:
            writer = csv.DictWriter(g, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in reader:
                for c in goal_names:
                    r[c] = "?"
                writer.writerow(r)


def _solve(A, b):
    """Gauss-Jordan elimination with partial pivoting for a square system."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if p == 0.0:
            continue
        for r in range(n):
            if r != col:
                f = M[r][col] / p
                if f:
                    for c in range(col, n + 1):
                        M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] if M[i][i] != 0.0 else 0.0 for i in range(n)]


def _ridge(X, y, p):
    """Ridge least squares weights for rows X (each of length p) and targets y."""
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for x, t in zip(X, y):
        for i in range(p):
            xi = x[i]
            b[i] += xi * t
            row = A[i]
            for j in range(p):
                row[j] += xi * x[j]
    for i in range(p):
        A[i][i] += RIDGE
    return _solve(A, b)


def _r2(ys, preds):
    if len(ys) < 2:
        return 0.0
    mean = sum(ys) / len(ys)
    sst = sum((y - mean) ** 2 for y in ys)
    if sst <= 0:
        return 0.0
    sse = sum((y - p) ** 2 for y, p in zip(ys, preds))
    return 1.0 - sse / sst


def race(train_csv, test_csv) -> dict[str, float]:
    """Score each method on test_csv (higher is better). Keys:
      'ours'               step7.choose, tuning on test builds, true score of the pick
      'random_search'      random benchmarks, true score of the pick
      'stdlib_hill_climb'  stdlib stand-in tuner, true score of the pick
      'oracle'             best true score in test_csv (upper bound)
      'ours_predict_r2'    step2.guess, mean R^2 over goals on test_csv
      'linear_predict_r2'  stdlib ridge regression, mean R^2 over goals
      plus per-goal prediction keys 'ours_predict_r2:<goal>' and
      'linear_predict_r2:<goal>'.
    """
    test_rows, inputs, goals = _load(test_csv)
    n = len(test_rows)
    if n == 0 or not goals:
        return {"ours": 0.0, "random_search": 0.0, "stdlib_hill_climb": 0.0,
                "oracle": 0.0, "ours_predict_r2": 0.0, "linear_predict_r2": 0.0}

    # Tuning race.
    truth = _true_scores(test_rows, goals)
    results = {"oracle": max((s for s in truth if s is not None), default=0.0)}
    budget = min(BUDGET, n)

    def label(i):
        return _goal_values(test_rows[i], goals)

    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _write_blind(test_csv, [c for c, _ in goals], blind)
        ours = choose(blind, label, budget=budget)

    results["ours"] = _truth_at(truth, ours)
    results["random_search"] = _truth_at(truth, _random_search(n, label, goals, budget))
    results["stdlib_hill_climb"] = _truth_at(
        truth, _hill_climb(test_rows, inputs, label, goals, budget))

    # Prediction race.
    train_rows, train_inputs, train_goals = _load(train_csv)
    train_names = {c for c, _ in train_goals}
    names = [c for c, _ in goals if c in train_names]
    ntr, nte = len(train_rows), n

    enc = _encode(train_rows + test_rows, train_inputs)
    Xall = [[1.0] + x for x in enc]
    Xtr, Xte = Xall[:ntr], Xall[ntr:]
    p = len(Xall[0]) if Xall else 1

    guess_preds = [guess(train_csv, {c: r.get(c) for c in train_inputs})
                   for r in test_rows]

    per_goal = {}
    for name in names:
        tr_x, tr_y = [], []
        for i, r in enumerate(train_rows):
            y = _finite(r.get(name))
            if y is not None:
                tr_x.append(Xtr[i])
                tr_y.append(y)
        if tr_y:
            w = _ridge(tr_x, tr_y, p)
            lin = [sum(a * b for a, b in zip(w, x)) for x in Xte]
        else:
            lin = [0.0] * nte

        idx, ys = [], []
        for i, r in enumerate(test_rows):
            y = _finite(r.get(name))
            if y is not None:
                idx.append(i)
                ys.append(y)
        g_pred = [guess_preds[i][name][0] for i in idx]
        l_pred = [lin[i] for i in idx]
        per_goal[name] = (_r2(ys, g_pred), _r2(ys, l_pred))

    if per_goal:
        g_vals = [g for g, _ in per_goal.values()]
        l_vals = [l for _, l in per_goal.values()]
        results["ours_predict_r2"] = sum(g_vals) / len(g_vals)
        results["linear_predict_r2"] = sum(l_vals) / len(l_vals)
        for name, (g, l) in per_goal.items():
            results[f"ours_predict_r2:{name}"] = g
            results[f"linear_predict_r2:{name}"] = l
    else:
        results["ours_predict_r2"] = 0.0
        results["linear_predict_r2"] = 0.0

    return results

