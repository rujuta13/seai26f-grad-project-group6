"""Step 20: does our shortlist beat a simple 'top overall rating' list, and does
our penalty guesser beat an off-the-shelf model?

race(train_csv, test_csv) returns {method: score}. Higher is better for every
key. A method whose score cannot be computed is reported as -inf.

Shortlist race. The test builds form a blind pool, and each method may label
at most _BUDGET of them. The score is the chosen build's mean goal score over
the complete test builds, each goal scaled over those builds (0 worst, 1 best).
  "ours"                step17.choose (scout-limited step7 search)
  "top_overall_rating"  label the _BUDGET builds with the highest OVR and keep
                        the best labelled one (the simple rating list)
  "random_search"       label random builds, keep the best labelled one
  "off_the_shelf_model" ridge-guided labelling, as in step10

Penalty guesser race. Only the Penalties goal is predicted, on test rows where
the Penalties value is finite. The score is minus the mean absolute error,
scaled by the Penalties range in train_csv.
  "penalty_guesser_neg_mae"  step12.guess median estimate ("ours")
  "penalty_ridge_neg_mae"    ridge regression fitted on train_csv
"""
import math
import os
import tempfile

from step1 import _load
from step10 import (
    _BUDGET,
    _Encoder,
    _best_of,
    _combined,
    _complete,
    _finite,
    _goal_specs,
    _is_goal,
    _is_input,
    _model_search,
    _neg_mae,
    _random_search,
    _ridge_preds,
    _spans,
    _write_blind,
)
from step12 import guess as _guess12
from step17 import choose as _choose17


def _top_ovr(pool, vecs, budget):
    """Label the budget builds with the highest OVR and return the best labelled one.

    Builds with no finite OVR are ranked last; ties go to the lower index.
    """
    def ovr(i):
        v = _finite(pool[i].get("OVR"))
        return v if v is not None else -math.inf

    ids = sorted(range(len(pool)), key=lambda i: (-ovr(i), i))[:budget]
    return _best_of(ids, vecs)


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores (higher is better) for our shortlist and penalty guesser and the rivals.

    Keys: "ours", "top_overall_rating", "random_search", "off_the_shelf_model"
    (shortlist race, score = mean goal score of the chosen build) and
    "penalty_guesser_neg_mae", "penalty_ridge_neg_mae" (Penalties prediction
    race, score = minus the mean scaled absolute error).
    """
    header, train = _load(train_csv)
    theader, test = _load(test_csv)
    specs = _goal_specs(theader)
    train_inputs = [c for c in header if _is_input(c)]
    test_inputs = [c for c in theader if _is_input(c)]

    scores = {
        "ours": -math.inf,
        "top_overall_rating": -math.inf,
        "random_search": -math.inf,
        "off_the_shelf_model": -math.inf,
        "penalty_guesser_neg_mae": -math.inf,
        "penalty_ridge_neg_mae": -math.inf,
    }

    # Shortlist race on the complete test builds.
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
            ours = _choose17(blind, label, budget=budget)
        scores["ours"] = float(pool_score[ours])

        scores["random_search"] = float(
            pool_score[_random_search(n, vecs, budget)]
        )

        enc = _Encoder(pool, test_inputs)
        X = [enc.encode(r) for r in pool]
        scores["off_the_shelf_model"] = float(
            pool_score[_model_search(X, vecs, budget)]
        )

        if "OVR" in theader:
            scores["top_overall_rating"] = float(
                pool_score[_top_ovr(pool, vecs, budget)]
            )

    # Penalty guesser race on the Penalties goal.
    pen_cols = [c for c in header if _is_goal(c) and c[:-1] == "Penalties"]
    if pen_cols and pen_cols[0] in theader and train and test:
        pen = pen_cols[0]
        spans = _spans(train, [pen])

        g_preds = []
        for r in test:
            row = {c: r[c] for c in train_inputs if c in r}
            est = _guess12(train_csv, row)
            g_preds.append([est[pen][0] if pen in est else None])
        scores["penalty_guesser_neg_mae"] = float(
            _neg_mae(g_preds, test, [pen], spans)
        )

        r_preds = _ridge_preds(train, test, train_inputs, [pen])
        scores["penalty_ridge_neg_mae"] = float(
            _neg_mae(r_preds, test, [pen], spans)
        )

    return scores

