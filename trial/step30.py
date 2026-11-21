"""Step 30: race our targeting and response guesser against simple rivals.

race() runs step10's tuner and guesser comparison on test_csv and adds two
client-specific rivals:

* "top_income": the regret of mailing the single highest-INCOME customer
  (ties go to the earlier row), scored the same way as step10's regret.
* Response-only guesser comparison: "guesser_ours_response_nmae" (step2's
  guess() for RESPONSE+) against "guesser_ridge_response_nmae" (a stdlib ridge
  regression for RESPONSE+), each as mean absolute error divided by the
  RESPONSE+ range over test_csv.

Every score is lower-is-better, and "ours" is always present.
"""

from step1 import _load, _split_columns
from step2 import guess
from step10 import _features, _finite, _fit_all, _normalised_scores
from step10 import race as _race_step10

RESPONSE_GOAL = "RESPONSE+"


def _top_income_regret(test_csv):
    """Regret of the test customer with the highest INCOME, or 1.0 if none qualifies."""
    columns, test = _load(test_csv)
    maximise, minimise, _ = _split_columns(columns)
    if "INCOME" not in columns:
        return 1.0

    scores = _normalised_scores(test, maximise, minimise)
    valid = [i for i, s in enumerate(scores) if s is not None]
    if not valid:
        return 1.0
    best_score = min(scores[i] for i in valid)

    candidates = [i for i in valid if _finite(test[i].get("INCOME")) is not None]
    if not candidates:
        return 1.0
    pick = min(candidates, key=lambda i: (-_finite(test[i].get("INCOME")), i))
    return float(scores[pick] - best_score)


def _response_errors(train_csv, test_csv):
    """Return (guess_nmae, ridge_nmae) for RESPONSE+ alone; (1.0, 1.0) if not computable."""
    train_columns, train = _load(train_csv)
    _, _, train_inputs = _split_columns(train_columns)
    test_columns, test = _load(test_csv)

    if RESPONSE_GOAL not in train_columns or RESPONSE_GOAL not in test_columns:
        return 1.0, 1.0

    spec, models = _fit_all(train, train_inputs, [RESPONSE_GOAL])
    if RESPONSE_GOAL not in models:
        return 1.0, 1.0

    known = [v for v in (_finite(r.get(RESPONSE_GOAL)) for r in test) if v is not None]
    if not known:
        return 1.0, 1.0
    span = max(known) - min(known)
    scale = span if span > 0 else 1.0

    ours_sum = 0.0
    ridge_sum = 0.0
    count = 0
    for r in test:
        truth = _finite(r.get(RESPONSE_GOAL))
        if truth is None:
            continue
        probe = {c: r.get(c) for c in train_inputs}
        est = guess(train_csv, probe)
        if RESPONSE_GOAL not in est:
            continue
        feats = _features(probe, spec)
        pred = sum(w * f for w, f in zip(models[RESPONSE_GOAL], feats))
        ours_sum += abs(est[RESPONSE_GOAL][0] - truth) / scale
        ridge_sum += abs(pred - truth) / scale
        count += 1

    if count == 0:
        return 1.0, 1.0
    return ours_sum / count, ridge_sum / count


def race(train_csv, test_csv) -> dict[str, float]:
    """Return {method: score} on test_csv. Every score is lower-is-better.

    Includes "ours" (step7's choose() regret), the tuner rivals "random_search"
    and "local_search", and the overall guesser errors "guesser_ours_nmae" and
    "guesser_ridge_nmae" from step10. Adds "top_income", the regret of the
    highest-income customer, and the RESPONSE+-only guesser errors
    "guesser_ours_response_nmae" and "guesser_ridge_response_nmae".
    """
    result = {k: float(v) for k, v in _race_step10(train_csv, test_csv).items()}
    result["top_income"] = float(_top_income_regret(test_csv))

    ours_response, ridge_response = _response_errors(train_csv, test_csv)
    result["guesser_ours_response_nmae"] = float(ours_response)
    result["guesser_ridge_response_nmae"] = float(ridge_response)
    return result

