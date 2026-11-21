"""Step 30: race. Two client questions, answered on test_csv.

  - "Does our targeting beat simply mailing our highest-income customers?"
  - "Does our response guesser beat an off-the-shelf model?"

Targeting race. The mailing list is the MAIL_FRAC share of test rows (at least one).
The score is the mean scaled value of the target goal (RESPONSE+ when the file has it,
otherwise the first goal present in both files) over the mailed customers. Each goal
value is min-max scaled over the test customers (0 = worst, 1 = best). Unusable
cells count as 0. For RESPONSE+ this is the response rate of the mailed group.
  - ours: rank customers by the target estimate from step22.guess, using inputs only.
    The test goal cells are never read for ranking.
  - top_income: the customers with the highest INCOME (file order if there is no
    INCOME column).
  - random_targeting: a seeded random mailing list.
  - oracle_targeting: the true best mailing list (an upper bound, since it reads the
    test goals).

Response race. Scores are R^2 on test_csv for the target goal, higher is better.
  - ours_response_r2: the step22.guess point estimate.
  - ridge_response_r2: stdlib ridge regression on the numeric inputs (standardised,
    median-filled) and one-hot text inputs, fitted on train_csv. This is an
    off-the-shelf model.
  - ours_response_coverage: the share of judged test customers whose true target
    value lies inside the guess low..high range.

All returned values are higher-is-better. "ours" is the targeting score.
"""
import math
import random

from step1 import _load
from step22 import guess

SEED = 1                  # fixed seed so results are reproducible
MAIL_FRAC = 0.10          # share of test customers mailed in the targeting race
TARGET_GOAL = "RESPONSE+"
NUMERIC_FRAC = 0.9        # a column is numeric if this share of its cells parse as numbers
ID_FRAC = 0.9             # numeric column with more distinct values than this share is an id
MAX_LEVELS = 25           # text columns with more levels are identifier-like and skipped
RIDGE = 1.0               # ridge penalty on the non-intercept weights
INCOME = "INCOME"

KEYS = ("ours", "random_targeting", "top_income", "oracle_targeting",
        "ours_response_r2", "ridge_response_r2", "ours_response_coverage")


def _finite(v):
    """The cell as a finite float, or None if it is missing, non-numeric or non-finite."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v.strip()
    return "" if v is None else str(v).strip()


def _zeros():
    return {k: 0.0 for k in KEYS}


def _plan(train_rows, inputs):
    """Decide how each input column is encoded, from training rows.

    Returns (num, cat):
      num  list of (column, mean, sd, median) for standardised numeric columns.
      cat  list of (column, sorted levels) for text columns with few levels.
    """
    n = len(train_rows)
    num, cat = [], []
    for c in inputs:
        raw = [r.get(c) for r in train_rows]
        vals = [_finite(v) for v in raw]
        present = [x for x in vals if x is not None]
        if present and len(present) >= NUMERIC_FRAC * n:
            if n >= 20 and len(set(present)) > ID_FRAC * n:
                continue  # identifier-like numeric column
            srt = sorted(present)
            med = srt[len(srt) // 2]
            filled = [x if x is not None else med for x in vals]
            mean = sum(filled) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in filled) / n)
            if sd > 0:
                num.append((c, mean, sd, med))
        else:
            levels = sorted({_text(v) for v in raw} - {""})
            if 2 <= len(levels) <= MAX_LEVELS:
                cat.append((c, levels))
    return num, cat


def _design(r, num, cat):
    """Feature vector for one row, intercept first."""
    x = [1.0]
    for c, mean, sd, med in num:
        v = _finite(r.get(c))
        if v is None:
            v = med
        x.append((v - mean) / sd)
    for c, levels in cat:
        s = _text(r.get(c))
        x.extend(1.0 if s == lv else 0.0 for lv in levels)
    return x


def _solve(A, b):
    """Gaussian elimination with partial pivoting for a square system."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if abs(p) < 1e-12:
            continue
        for r in range(col + 1, n):
            f = M[r][col] / p
            if f:
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    w = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n] - sum(M[i][j] * w[j] for j in range(i + 1, n))
        w[i] = s / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0
    return w


def _ridge(X, y, p):
    """Ridge least squares. The intercept (index 0) is not penalised."""
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for x, t in zip(X, y):
        for i in range(p):
            xi = x[i]
            if xi == 0.0:
                continue
            b[i] += xi * t
            row = A[i]
            for j in range(p):
                row[j] += xi * x[j]
    for i in range(1, p):
        A[i][i] += RIDGE
    return _solve(A, b)


def _r2(ys, preds):
    """R^2 of predictions against true values (0.0 if undefined)."""
    if len(ys) < 2:
        return 0.0
    mean = sum(ys) / len(ys)
    sst = sum((y - mean) ** 2 for y in ys)
    if sst <= 0:
        return 0.0
    sse = sum((y - p) ** 2 for y, p in zip(ys, preds))
    return 1.0 - sse / sst


def race(train_csv, test_csv) -> dict[str, float]:
    """Score the targeting and response methods on test_csv (higher is better).

    Returns a dict with keys 'ours' (targeting score of our mailing list),
    'random_targeting', 'top_income', 'oracle_targeting', 'ours_response_r2',
    'ridge_response_r2' and 'ours_response_coverage'.
    """
    train_rows, train_inputs, train_goals = _load(train_csv)
    test_rows, test_inputs, test_goals = _load(test_csv)
    n = len(test_rows)
    if n == 0 or not train_rows or not train_goals or not test_goals:
        return _zeros()

    train_names = {c: s for c, s in train_goals}
    test_names = {c: s for c, s in test_goals}
    if TARGET_GOAL in train_names and TARGET_GOAL in test_names:
        target = TARGET_GOAL
    else:
        target = next((c for c, _ in train_goals if c in test_names), None)
    if target is None:
        return _zeros()
    sign = train_names[target]

    # True target values on test_csv, raw and scaled to 0..1 (higher is better).
    ys = [_finite(r.get(target)) for r in test_rows]
    oriented = [None if y is None else sign * y for y in ys]
    usable = [o for o in oriented if o is not None]
    lo = min(usable) if usable else 0.0
    hi = max(usable) if usable else 0.0
    span = hi - lo
    truth = []
    for o in oriented:
        if o is None or span <= 0:
            truth.append(0.0)
        else:
            truth.append((o - lo) / span)

    # Guesses from inputs only.
    input_rows = [{c: r.get(c) for c in train_inputs} for r in test_rows]
    ests = [guess(train_csv, row)[target] for row in input_rows]
    est_val = [e[0] for e in ests]
    est_or = [sign * v for v in est_val]

    # Mailing lists.
    k = max(1, min(n, int(MAIL_FRAC * n)))

    def mean_truth(picks):
        return sum(truth[i] for i in picks) / len(picks) if picks else 0.0

    ours_pick = sorted(range(n), key=lambda i: (-est_or[i], i))[:k]
    oracle_pick = sorted(range(n), key=lambda i: (-truth[i], i))[:k]
    random_pick = random.Random(SEED).sample(range(n), k)

    if INCOME in test_inputs:
        inc = [_finite(r.get(INCOME)) for r in test_rows]
        income_pick = sorted(
            range(n),
            key=lambda i: (math.inf if inc[i] is None else -inc[i], i),
        )[:k]
    else:
        income_pick = list(range(n))[:k]

    # Response race: ridge regression fitted on train_csv.
    num, cat = _plan(train_rows, train_inputs)
    fit_x, fit_y = [], []
    for r in train_rows:
        y = _finite(r.get(target))
        if y is not None:
            fit_x.append(_design(r, num, cat))
            fit_y.append(y)
    p = 1 + len(num) + sum(len(lv) for _, lv in cat)
    if fit_y:
        w = _ridge(fit_x, fit_y, p)
    else:
        w = [0.0] * p
    ridge_pred = [sum(a * b for a, b in zip(w, _design(r, num, cat)))
                  for r in test_rows]

    judged = [i for i in range(n) if ys[i] is not None]
    yt = [ys[i] for i in judged]
    ours_r2 = _r2(yt, [est_val[i] for i in judged])
    ridge_r2 = _r2(yt, [ridge_pred[i] for i in judged])

    covered = 0
    ranged = 0
    for i in judged:
        _, low, high = ests[i]
        if math.isfinite(low) and math.isfinite(high):
            ranged += 1
            if low <= ys[i] <= high:
                covered += 1
    coverage = covered / ranged if ranged else 0.0

    return {
        "ours": float(mean_truth(ours_pick)),
        "random_targeting": float(mean_truth(random_pick)),
        "top_income": float(mean_truth(income_pick)),
        "oracle_targeting": float(mean_truth(oracle_pick)),
        "ours_response_r2": float(ours_r2),
        "ridge_response_r2": float(ridge_r2),
        "ours_response_coverage": float(coverage),
    }

