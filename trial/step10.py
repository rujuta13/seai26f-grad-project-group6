"""Step 10: race. Our tuner and guesser against standard-library rivals on test data.

Two races, both scored on test_csv. Every score lies in [0, 1] and higher is better.

Tuning (budget of labels per method, mirroring step7.choose):
  - "ours": step7.choose.
  - "random_search": label _BUDGET test rows drawn uniformly at random.
  - "local_search": start at a random row, then repeatedly label the unlabelled
    row nearest to the current best (a greedy nearest-neighbour hill climb).
  Score: the chosen row's normalised goal total over all test rows, divided by
  the number of goals (1 means best possible on every goal). The randomised
  rivals use several seeds and report the mean.

Prediction (goal values from the flag set alone):
  - "ours_guesser": step2.guess point estimate (median of nearest neighbours).
  - "linear_model": ridge regression fitted on train_csv, solved in closed form
    with Gaussian elimination.
  Score per goal: 1 - MAE / (training range of the goal), clipped to [0, 1];
  averaged over the goals that have enough training values.

The standard library ships no tuner and no off-the-shelf regression model, so
both rivals are written here using only the standard library.
"""

import math
import random

import step1
import step2
import step7

_BUDGET = 30                # label budget per tuning method (as step7.choose default)
_SEEDS = (1, 2, 3, 4, 5)    # seeds for the randomised rivals; their scores are averaged
_RIDGE = 1e-3               # ridge penalty for the linear model (applied to all coefficients)
_EPS = 1e-12


def _zeros():
    return {"ours": 0.0, "random_search": 0.0, "local_search": 0.0,
            "ours_guesser": 0.0, "linear_model": 0.0}


def _measure(rows, goals, i):
    """Goal values of row i as {goal: float or None}."""
    return {g: step7._num(rows[i].get(g)) for g, _ in goals}


def _labeller(rows, goals):
    """Label function for step7.choose: returns the goals of row i from the test data."""
    def label(i):
        return {g: rows[i].get(g) for g, _ in goals}
    return label


def _best_measured(measured, goals):
    """Best measured row index (ties to the lower index), as step7 does."""
    totals = step7._totals(measured, goals)
    return min(measured, key=lambda i: (-totals[i], i))


def _random_search(rows, goals, seed):
    rng = random.Random(seed)
    picks = rng.sample(range(len(rows)), min(_BUDGET, len(rows)))
    measured = {i: _measure(rows, goals, i) for i in picks}
    return _best_measured(measured, goals)


def _local_search(rows, feats, goals, seed):
    n = len(rows)
    rng = random.Random(seed)
    budget = min(_BUDGET, n)
    start = rng.randrange(n)
    measured = {start: _measure(rows, goals, start)}
    best = start
    while len(measured) < budget:
        cands = [j for j in range(n) if j not in measured]
        nxt = min(cands, key=lambda j: (step1._dist2(feats[j], feats[best]), j))
        measured[nxt] = _measure(rows, goals, nxt)
        best = _best_measured(measured, goals)
    return best


def _tuning(test_csv, test_rows, inputs, goals):
    n = len(test_rows)
    full = {i: _measure(test_rows, goals, i) for i in range(n)}
    totals = step7._totals(full, goals)  # normalised over all test rows

    def score(i):
        return float(totals[i] / len(goals))

    feats = step1._features(test_rows, inputs)

    ours = step7.choose(test_csv, _labeller(test_rows, goals), budget=_BUDGET)
    out = {"ours": score(ours)}
    out["random_search"] = (
        sum(score(_random_search(test_rows, goals, s)) for s in _SEEDS) / len(_SEEDS)
    )
    out["local_search"] = (
        sum(score(_local_search(test_rows, feats, goals, s)) for s in _SEEDS)
        / len(_SEEDS)
    )
    return out


def _fit_solve(A, b):
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < _EPS:
            continue
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / p
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) >= _EPS else 0.0 for i in range(n)]


def _fit_ridge(X, y):
    """Closed-form ridge regression weights for rows X (already with a bias term) and targets y."""
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for x, t in zip(X, y):
        for i in range(d):
            xi = x[i]
            b[i] += xi * t
            Ai = A[i]
            for j in range(d):
                Ai[j] += xi * x[j]
    for i in range(d):
        A[i][i] += _RIDGE
    return _fit_solve(A, b)


def _ridge_predictions(train_rows, test_rows, inputs, goals_used):
    """{goal: predictions for every test row} from a ridge model fitted on train_rows."""
    feats = step1._features(train_rows + test_rows, inputs)
    ntr = len(train_rows)
    X = [list(f) + [1.0] for f in feats]
    preds = {}
    for g in goals_used:
        pairs = []
        for i in range(ntr):
            t = step7._num(train_rows[i].get(g))
            if t is not None:
                pairs.append((X[i], t))
        xs = [x for x, _ in pairs]
        ys = [t for _, t in pairs]
        ym = sum(ys) / len(ys)
        sd = math.sqrt(sum((t - ym) ** 2 for t in ys) / len(ys)) or 1.0
        w = _fit_ridge(xs, [(t - ym) / sd for t in ys])
        preds[g] = [
            ym + sd * sum(a * b for a, b in zip(w, X[ntr + j]))
            for j in range(len(test_rows))
        ]
    return preds


def _goal_score(preds, actuals, span):
    """1 - MAE / span, clipped to [0, 1]. Missing or non-finite predictions count as span."""
    errs = 0.0
    got = 0
    for p, a in zip(preds, actuals):
        if a is None:
            continue
        got += 1
        if math.isfinite(p):
            errs += abs(p - a)
        else:
            errs += span
    if got == 0:
        return None
    mae = errs / got
    if span <= 0:
        return 1.0 if mae <= _EPS else 0.0
    return max(0.0, min(1.0, 1.0 - mae / span))


def _prediction(train_csv, train_rows, test_rows, inputs, goals):
    spans = {}
    usable = []
    for g, _ in goals:
        vals = [v for v in (step7._num(r.get(g)) for r in train_rows) if v is not None]
        if len(vals) >= 2:
            usable.append(g)
            spans[g] = max(vals) - min(vals)
    if not usable:
        return {"ours_guesser": 0.0, "linear_model": 0.0}

    actual = {g: [step7._num(r.get(g)) for r in test_rows] for g in usable}

    guessed = {g: [] for g in usable}
    for r in test_rows:
        flags = {n: r[n] for n in inputs if n in r}
        est = step2.guess(train_csv, flags)
        for g in usable:
            guessed[g].append(float(est.get(g, (math.nan,))[0]))

    ridge = _ridge_predictions(train_rows, test_rows, inputs, usable)

    def average(preds):
        per = []
        for g in usable:
            s = _goal_score(preds[g], actual[g], spans[g])
            if s is not None:
                per.append(s)
        return float(sum(per) / len(per)) if per else 0.0

    return {"ours_guesser": average(guessed), "linear_model": average(ridge)}


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our tuner and guesser against stdlib-only rivals on test_csv.

    Returns {method: score}, higher is better, all in [0, 1]:
      "ours", "random_search", "local_search"  - tuning, normalised goal quality
      "ours_guesser", "linear_model"           - prediction, 1 - MAE / range
    """
    header, test_rows = step1._read(test_csv)
    _, train_rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not test_rows or not goals:
        return _zeros()

    results = {}
    results.update(_tuning(test_csv, test_rows, inputs, goals))
    results.update(_prediction(train_csv, train_rows, test_rows, inputs, goals))
    return results

