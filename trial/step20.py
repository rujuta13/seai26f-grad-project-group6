import math
import random

import step1
import step10
import step12

_SEED = 0            # fixed seed so the same inputs always give the same answer
_SHORTLIST = 0.05    # shortlist size as a share of the rows (same cap as step1.best)
_RANDOM_DRAWS = 200  # random shortlists averaged for the random baseline
_MAX_CATS = 10       # categorical inputs with more values are left out of the model
_OVR_NAME = "ovr"    # overall-rating input, matched case-insensitively
_PENALTY_HINT = "penal"


def _size(n):
    return max(1, int(_SHORTLIST * n)) if n > 0 else 0


def _shortlist_quality(shortlist, signed):
    """For each goal, take the best value inside the shortlist and measure the
    share of rows it matches or beats. Average over goals (0..1)."""
    n = len(signed)
    if n == 0 or not shortlist or not signed[0]:
        return 0.0
    g = len(signed[0])
    total = 0.0
    for k in range(g):
        best = max(signed[i][k] for i in shortlist)
        total += sum(1 for t in signed if t[k] <= best) / n
    return total / g


def _top_by_rank(signed, size):
    """Shortlist by the highest sum of per-goal ranks (same idea as step1.best)."""
    n = len(signed)
    if n == 0 or size <= 0:
        return []
    rank_sum = [0] * n
    for k in range(len(signed[0])):
        order = sorted(range(n), key=lambda i: (signed[i][k], i))
        for pos, i in enumerate(order):
            rank_sum[i] += pos
    order = sorted(range(n), key=lambda i: (-rank_sum[i], i))
    return order[:size]


def _top_by_column(rows, col, size):
    """Shortlist by one rating column, highest first; missing values rank last."""
    vals = [step10._finite_or_none(r.get(col, "")) for r in rows]
    order = sorted(
        range(len(rows)),
        key=lambda i: (-vals[i] if vals[i] is not None else math.inf, i),
    )
    return order[:size]


def _ovr_column(inputs):
    for c in inputs:
        if c.strip().lower() == _OVR_NAME:
            return c
    return None


def _penalty_goal(goals):
    for name, _sign in goals:
        if _PENALTY_HINT in name.lower():
            return name
    return goals[0][0] if goals else None


def _ridge_spec(trows, inputs):
    """Standardised numeric inputs, plus one-hot inputs with few categories.
    Columns with many distinct values (names, links) are left out, which
    keeps the normal equations small."""
    spec = []
    for c in inputs:
        nums = [step10._finite_or_none(r.get(c, "")) for r in trows]
        if nums and all(x is not None for x in nums):
            mean = sum(nums) / len(nums)
            var = sum((x - mean) ** 2 for x in nums) / len(nums)
            sd = math.sqrt(var) if var > 0 else 1.0
            spec.append(("num", c, mean, sd))
        else:
            cats = sorted({str(r.get(c, "")) for r in trows})
            if len(cats) <= _MAX_CATS:
                spec.append(("cat", c, cats))
    return spec


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare our shortlist and penalty guesser with simple rivals on test_csv.

    Shortlist keys are on the same 0..1 scale: the share of test rows that the
    shortlist's best value on each goal matches or beats, averaged over goals.
      "ours"                  shortlist from step12 guesses made from inputs
                              only, using train_csv (no test goal values used).
      "top_overall_rating"    the rows with the highest OVR rating (0.0 if the
                              file has no OVR column).
      "random_shortlist"      random shortlists of the same size, averaged.
      "hindsight_shortlist"   step1.best on test_csv's own goals. This is an
                              upper reference, not a fair competitor.

    Penalty keys are skill scores on the same scale as step10: 1 minus the
    model's mean absolute error divided by the error of predicting the training
    median, so 1 is perfect, 0 is no better than the median, and negative is worse.
      "ours_penalty_guesser"  step12 nearest-neighbour guess of the penalty goal.
      "stdlib_penalty_model"  ridge regression on the same goal.
    """
    keys = (
        "ours",
        "top_overall_rating",
        "random_shortlist",
        "hindsight_shortlist",
        "ours_penalty_guesser",
        "stdlib_penalty_model",
    )
    zero = {k: 0.0 for k in keys}

    test_rows, test_inputs, test_goals = step1._load(test_csv)
    n = len(test_rows)
    if n == 0 or not test_goals:
        return zero

    trows, inputs, _train_goals = step1._load(train_csv)
    if not trows:
        return zero

    signed = [step10._signed(r, test_goals) for r in test_rows]
    size = _size(n)

    # ours: guesses made from inputs only, ranked to form the shortlist
    guesses = [step12.guess(train_csv, r) for r in test_rows]
    predicted = []
    for g in guesses:
        vals = []
        for name, sign in test_goals:
            est = g.get(name)
            if est is not None and math.isfinite(est[0]):
                vals.append(sign * est[0])
            else:
                vals.append(-math.inf)
        predicted.append(tuple(vals))
    ours = _shortlist_quality(_top_by_rank(predicted, size), signed)

    # rival: top overall rating list
    ovr = _ovr_column(test_inputs)
    if ovr is not None:
        top_overall = _shortlist_quality(
            _top_by_column(test_rows, ovr, size), signed)
    else:
        top_overall = 0.0

    # rival: random shortlists of the same size
    rng = random.Random(_SEED)
    total = 0.0
    for _ in range(_RANDOM_DRAWS):
        total += _shortlist_quality(rng.sample(range(n), min(size, n)), signed)
    random_shortlist = total / _RANDOM_DRAWS

    # reference: shortlist chosen with the test file's own goals
    hindsight = _shortlist_quality(step1.best(test_csv), signed)

    # penalty guesser vs off-the-shelf ridge model
    target = _penalty_goal(test_goals)
    truth = [step10._finite_or_none(r.get(target, "")) for r in test_rows] \
        if target is not None else []
    eval_idx = [i for i, v in enumerate(truth) if v is not None]
    train_vals = [v for v in (step10._finite_or_none(r.get(target, ""))
                              for r in trows) if v is not None] \
        if target is not None else []

    ours_penalty = 0.0
    stdlib_penalty = 0.0
    if eval_idx and train_vals:
        med = step10._median(train_vals)
        base_mae = sum(abs(truth[i] - med) for i in eval_idx) / len(eval_idx)

        g_err = 0.0
        for i in eval_idx:
            est = guesses[i].get(target)
            pred_v = est[0] if est is not None and math.isfinite(est[0]) else med
            g_err += abs(truth[i] - pred_v)
        ours_penalty = step10._skill(g_err / len(eval_idx), base_mae)

        spec = _ridge_spec(trows, inputs)
        X, y = [], []
        for r in trows:
            v = step10._finite_or_none(r.get(target, ""))
            if v is not None:
                X.append(step10._features(r, spec))
                y.append(v)
        if X:
            w = step10._fit(X, y)
            m_err = 0.0
            for i in eval_idx:
                feats = step10._features(test_rows[i], spec)
                pred_v = sum(a * b for a, b in zip(w, feats))
                if not math.isfinite(pred_v):
                    pred_v = med
                m_err += abs(truth[i] - pred_v)
            stdlib_penalty = step10._skill(m_err / len(eval_idx), base_mae)

    return {
        "ours": float(ours),
        "top_overall_rating": float(top_overall),
        "random_shortlist": float(random_shortlist),
        "hindsight_shortlist": float(hindsight),
        "ours_penalty_guesser": float(ours_penalty),
        "stdlib_penalty_model": float(stdlib_penalty),
    }

