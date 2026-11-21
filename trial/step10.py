import math
import random

import step1
import step2
import step7

_SEED = 0            # fixed seed so the same inputs always give the same answer
_BUDGET = 30         # labels allowed per tuner (same default as step7.choose)
_RIDGE = 1.0         # ridge penalty for the off-the-shelf linear model
# The standard library has no tuner, so random search is the tuning rival.
# The standard library has no ML model either, so the off-the-shelf rival is
# a ridge linear regression written here with only the standard library.
_SPEED_HINTS = ("cpu", "time", "speed", "build", "compile")


def _signed(row, goals):
    """Goal values signed so higher is better; missing values count as worst."""
    out = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        out.append(sign * v if v is not None else -math.inf)
    return tuple(out)


def _quality(chosen, signed):
    """Mean over goals of the share of rows the chosen row matches or beats (0..1)."""
    n = len(signed)
    if n == 0:
        return 0.0
    g = len(signed[0])
    if g == 0:
        return 0.0
    c = signed[chosen]
    total = 0.0
    for k in range(g):
        total += sum(1 for t in signed if t[k] <= c[k]) / n
    return total / g


def _target(goals):
    """The goal that stands for build speed: a CPU/time-like goal if present."""
    if not goals:
        return None
    for name, _sign in goals:
        low = name.lower()
        if any(h in low for h in _SPEED_HINTS):
            return name
    return goals[0][0]


def _median(xs):
    s = sorted(xs)
    m = len(s)
    mid = m // 2
    if m % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _finite_or_none(v):
    x = step1._to_num(v)
    if x is None or not math.isfinite(x):
        return None
    return x


def _fit_spec(trows, inputs):
    """Per input: standardised numeric, or one-hot over the training categories."""
    spec = []
    for c in inputs:
        nums = [_finite_or_none(r.get(c, "")) for r in trows]
        if nums and all(x is not None for x in nums):
            mean = sum(nums) / len(nums)
            var = sum((x - mean) ** 2 for x in nums) / len(nums)
            sd = math.sqrt(var) if var > 0 else 1.0
            spec.append(("num", c, mean, sd))
        else:
            cats = sorted({str(r.get(c, "")) for r in trows})
            spec.append(("cat", c, cats))
    return spec


def _features(row, spec):
    x = [1.0]  # intercept
    for s in spec:
        if s[0] == "num":
            _, c, mean, sd = s
            v = _finite_or_none(row.get(c, ""))
            if v is None:
                v = mean
            x.append((v - mean) / sd)
        else:
            _, c, cats = s
            v = str(row.get(c, ""))
            x.extend(1.0 if v == k else 0.0 for k in cats)
    return x


def _solve(A, b):
    """Solve A w = b by Gauss-Jordan elimination with partial pivoting."""
    m = len(b)
    M = [A[i][:] + [b[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for r in range(m):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / p
                for k in range(col, m + 1):
                    M[r][k] -= f * M[col][k]
    w = [0.0] * m
    for i in range(m):
        if abs(M[i][i]) > 1e-12:
            w[i] = M[i][m] / M[i][i]
    return w


def _fit(X, y):
    """Ridge regression via the normal equations; intercept is not penalised."""
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for xi, yi in zip(X, y):
        for p in range(d):
            b[p] += xi[p] * yi
            row = A[p]
            for q in range(d):
                row[q] += xi[p] * xi[q]
    for p in range(1, d):
        A[p][p] += _RIDGE
    return _solve(A, b)


def _skill(mae, base):
    """1 - model error / error of predicting the training median (higher is better)."""
    if base <= 1e-12:
        return 0.0
    return 1.0 - mae / base


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our tuner and guesser against standard rivals on test_csv.

    Keys (higher is better on all):
      "ours"            step7 tuner: share of test rows the chosen row matches or
                        beats, averaged over goals (0..1).
      "random_search"   random rows labelled with the same budget, picked by the
                        same rule (same scale as "ours").
      "ours_guesser"    step2 guesser predicting the build-speed goal on test rows:
                        skill vs. predicting the training median (1 = perfect,
                        0 = no better than the median).
      "stdlib_model"    ridge linear regression on the same goal, same skill scale.
    """
    zero = {"ours": 0.0, "random_search": 0.0,
            "ours_guesser": 0.0, "stdlib_model": 0.0}

    test_rows, _test_inputs, test_goals = step1._load(test_csv)
    n = len(test_rows)
    if n == 0 or not test_goals:
        return zero

    signed = [_signed(r, test_goals) for r in test_rows]
    budget = _BUDGET

    # ours: step7 tuner, labels come from the test file's true goal values
    def label(i):
        return test_rows[i]

    ours_idx = step7.choose(test_csv, label, budget=budget)
    ours = _quality(ours_idx, signed)

    # random search: same budget, same final selection rule as step7
    rng = random.Random(_SEED)
    picks = rng.sample(range(n), min(budget, n))
    cache = {i: signed[i] for i in picks}
    rand_idx = step7._pick(cache, test_goals)
    random_search = _quality(rand_idx, signed)

    # predictors: build-speed goal, scored on test rows with a known value
    trows, inputs, train_goals = step1._load(train_csv)
    target = _target(train_goals)
    if target is None or not trows:
        return {"ours": ours, "random_search": random_search,
                "ours_guesser": 0.0, "stdlib_model": 0.0}

    truth = [_finite_or_none(r.get(target, "")) for r in test_rows]
    eval_idx = [i for i, v in enumerate(truth) if v is not None]
    train_vals = [v for v in (_finite_or_none(r.get(target, "")) for r in trows)
                  if v is not None]
    if not eval_idx or not train_vals:
        return {"ours": ours, "random_search": random_search,
                "ours_guesser": 0.0, "stdlib_model": 0.0}

    med = _median(train_vals)
    base_mae = sum(abs(truth[i] - med) for i in eval_idx) / len(eval_idx)

    # ours_guesser: step2 nearest-neighbour guess of the target
    g_err = 0.0
    for i in eval_idx:
        guessed = step2.guess(train_csv, test_rows[i]).get(target)
        est = guessed[0] if guessed is not None else med
        if not math.isfinite(est):
            est = med
        g_err += abs(truth[i] - est)
    ours_guesser = _skill(g_err / len(eval_idx), base_mae)

    # stdlib_model: ridge regression fitted on training rows with a known target
    spec = _fit_spec(trows, inputs)
    X, y = [], []
    for r in trows:
        v = _finite_or_none(r.get(target, ""))
        if v is None:
            continue
        X.append(_features(r, spec))
        y.append(v)
    if not X:
        stdlib_model = 0.0
    else:
        w = _fit(X, y)
        m_err = 0.0
        for i in eval_idx:
            feats = _features(test_rows[i], spec)
            pred = sum(a * b for a, b in zip(w, feats))
            if not math.isfinite(pred):
                pred = med
            m_err += abs(truth[i] - pred)
        stdlib_model = _skill(m_err / len(eval_idx), base_mae)

    return {
        "ours": float(ours),
        "random_search": float(random_search),
        "ours_guesser": float(ours_guesser),
        "stdlib_model": float(stdlib_model),
    }

