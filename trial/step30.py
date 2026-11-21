import math
import random

import step1
import step10
import step12
import step20

_SEED = 0  # fixed seed so the same inputs always give the same answer
_RESPONSE_HINTS = ("response", "acceptedcmp", "cmp")
_INCOME = "income"


def _income_column(inputs):
    """The input column named income, matched case-insensitively."""
    for c in inputs:
        if c.strip().lower() == _INCOME:
            return c
    for c in inputs:
        if _INCOME in c.lower():
            return c
    return None


def _response_goal(goals):
    """The goal that stands for campaign response, if one is named that way."""
    for hint in _RESPONSE_HINTS:
        for name, _sign in goals:
            if hint in name.lower():
                return name
    return goals[0][0] if goals else None


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare our targeting and response guesser with simple rivals on test_csv.

    Shortlist keys are on the same 0..1 scale: the share of test customers that
    the shortlist's best value on each goal matches or beats, averaged over goals.
      "ours"                  shortlist from step12 guesses made from inputs
                              only, using train_csv (no test goal values used).
      "top_income"            the customers with the highest INCOME (0.0 if the
                              file has no INCOME column). This is the client's
                              "mail our highest-income customers" rival.
      "random_shortlist"      random shortlists of the same size, averaged.
      "hindsight_shortlist"   step1.best on test_csv's own goals. An upper
                              reference, not a fair competitor.

    Response keys are skill scores on the step10 scale: 1 minus the model's mean
    absolute error divided by the error of predicting the training median, so 1
    is perfect, 0 is no better than the median, and negative is worse.
      "ours_response_guesser" step12 nearest-neighbour guess of the response goal.
      "stdlib_response_model" ridge regression on the same goal.
    """
    keys = (
        "ours",
        "top_income",
        "random_shortlist",
        "hindsight_shortlist",
        "ours_response_guesser",
        "stdlib_response_model",
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
    size = step20._size(n)

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
    ours = step20._shortlist_quality(step20._top_by_rank(predicted, size), signed)

    # rival: the highest-income customers
    inc = _income_column(test_inputs)
    if inc is not None:
        top_income = step20._shortlist_quality(
            step20._top_by_column(test_rows, inc, size), signed)
    else:
        top_income = 0.0

    # rival: random shortlists of the same size
    rng = random.Random(_SEED)
    total = 0.0
    for _ in range(step20._RANDOM_DRAWS):
        total += step20._shortlist_quality(
            rng.sample(range(n), min(size, n)), signed)
    random_shortlist = total / step20._RANDOM_DRAWS

    # reference: shortlist chosen with the test file's own goals
    hindsight = step20._shortlist_quality(step1.best(test_csv), signed)

    # response guesser vs off-the-shelf ridge model
    target = _response_goal(test_goals)
    truth = [step10._finite_or_none(r.get(target, "")) for r in test_rows]
    eval_idx = [i for i, v in enumerate(truth) if v is not None]
    train_vals = [v for v in (step10._finite_or_none(r.get(target, ""))
                              for r in trows) if v is not None]

    ours_guesser = 0.0
    stdlib_model = 0.0
    if eval_idx and train_vals:
        med = step10._median(train_vals)
        base_mae = sum(abs(truth[i] - med) for i in eval_idx) / len(eval_idx)

        g_err = 0.0
        for i in eval_idx:
            est = guesses[i].get(target)
            pred_v = est[0] if est is not None and math.isfinite(est[0]) else med
            g_err += abs(truth[i] - pred_v)
        ours_guesser = step10._skill(g_err / len(eval_idx), base_mae)

        spec = step20._ridge_spec(trows, inputs)
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
            stdlib_model = step10._skill(m_err / len(eval_idx), base_mae)

    return {
        "ours": float(ours),
        "top_income": float(top_income),
        "random_shortlist": float(random_shortlist),
        "hindsight_shortlist": float(hindsight),
        "ours_response_guesser": float(ours_guesser),
        "stdlib_response_model": float(stdlib_model),
    }

