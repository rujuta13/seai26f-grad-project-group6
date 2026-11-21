"""Step 20: race. Two client questions, answered on test_csv.

  - "Does our shortlist beat a simple top overall rating list?"
  - "Does our penalty guesser beat an off-the-shelf model?"

Shortlist race. Every method picks K players, where K is the step11 shortlist cap
(5% of test rows, at least one). The score is the mean step1-scale goal score of the
picked players on test_csv: each goal is min-max scaled over the test players with
complete goal values (0 = worst, 1 = best) and the scaled values are averaged.
  - ours: rank players by the goal score estimated by step12.guess from their inputs
    alone. The test goal cells are never read for ranking.
  - top_overall_rating: the K players with the highest OVR input column. If the file
    has no OVR column, the first K rows in file order.
  - random_shortlist: K seeded random players.
  - oracle: the true top K (an upper bound, since it reads the test goals).
The *_precision keys give the share of each shortlist that is in the true top K.

Penalty race. The goal is PENALTY_GOAL, or else the first goal present in both files.
Scores are R^2 on test_csv, higher is better.
  - ours_penalty_r2: the step12.guess point estimate.
  - linear_penalty_r2: stdlib ridge regression on the numeric inputs only (median-filled,
    standardised). Text columns and identifier-like numbers are not used, so this is a
    plain off-the-shelf linear model.
  - ours_penalty_coverage: the share of test players whose true value lies inside the
    guess low..high range. Only players with a finite range and a true value count.

All returned values are higher-is-better.
"""
import math
import random

from step1 import _load
from step10 import _finite, _r2, _ridge, _true_scores
from step12 import guess

SEED = 1                 # fixed seed so results are reproducible
SHORTLIST_FRAC = 0.05    # shortlist size as a share of test rows (as in step11.best)
RATING_COLUMN = "OVR"    # input column used as the top overall rating
PENALTY_GOAL = "Penalties+"
NUMERIC_FRAC = 0.9       # a column is numeric if this share of its cells parse as numbers
ID_FRAC = 0.9            # numeric column with more distinct values than this share is an id

_KEYS = ("ours", "top_overall_rating", "random_shortlist", "oracle",
         "ours_precision", "top_overall_rating_precision", "random_precision",
         "ours_penalty_r2", "linear_penalty_r2", "ours_penalty_coverage")


def _zero():
    return {k: 0.0 for k in _KEYS}


def _train_ranges(train_rows, goals):
    """Oriented (min, max) of each goal over training rows with a usable cell."""
    out = []
    for c, sign in goals:
        vals = [sign * y for y in (_finite(r.get(c)) for r in train_rows) if y is not None]
        out.append((min(vals), max(vals)) if vals else (0.0, 0.0))
    return out


def _est_score(est, goals, ranges):
    """Step1-style score of a guess() result, using only goals with a finite range.
    Returns None if no goal can be used."""
    total = 0.0
    used = 0
    for (c, sign), (lo, hi) in zip(goals, ranges):
        e = est.get(c)
        if e is None:
            continue
        value, low, high = e
        if not (math.isfinite(low) and math.isfinite(high)):
            continue
        used += 1
        span = hi - lo
        if span > 0:
            total += (sign * value - lo) / span
    return total / used if used else None


def _mean_truth(picks, truth):
    """Mean true score of the picked rows; unusable rows count as 0."""
    total = 0.0
    for i in picks:
        t = truth[i]
        if t is not None:
            total += t
    return total / len(picks) if picks else 0.0


def _precision(picks, top, k):
    """Share of the picks that are in the true top set."""
    return len(set(picks) & top) / k if k else 0.0


def _numeric_plan(rows, inputs):
    """(column, median, mean, sd) for each numeric input usable as a linear feature."""
    n = len(rows)
    plan = []
    for c in inputs:
        vals = [_finite(r.get(c)) for r in rows]
        present = [x for x in vals if x is not None]
        if not present or len(present) < NUMERIC_FRAC * n:
            continue
        if n >= 20 and len(set(present)) > ID_FRAC * n:
            continue  # identifier-like numeric column
        median = sorted(present)[len(present) // 2]
        filled = [x if x is not None else median for x in vals]
        mean = sum(filled) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in filled) / n)
        if sd > 0:
            plan.append((c, median, mean, sd))
    return plan


def _linear_features(row, plan):
    """Intercept followed by the standardised numeric inputs of one row."""
    x = [1.0]
    for c, median, mean, sd in plan:
        v = _finite(row.get(c))
        if v is None:
            v = median
        x.append((v - mean) / sd)
    return x


def _linear_predictions(train_rows, train_inputs, test_rows, pen):
    """Ridge regression of the penalty goal on numeric inputs, predicted for test_rows."""
    plan = _numeric_plan(train_rows, train_inputs)
    tr_x, tr_y = [], []
    for r in train_rows:
        y = _finite(r.get(pen))
        if y is not None:
            tr_x.append(_linear_features(r, plan))
            tr_y.append(y)
    if not tr_y:
        return [0.0] * len(test_rows)
    w = _ridge(tr_x, tr_y, 1 + len(plan))
    return [sum(a * b for a, b in zip(w, _linear_features(r, plan))) for r in test_rows]


def race(train_csv, test_csv) -> dict[str, float]:
    """Score the shortlist and penalty methods on test_csv (higher is better).

    Returns a dict with keys 'ours', 'top_overall_rating', 'random_shortlist', 'oracle',
    the '*_precision' keys, 'ours_penalty_r2', 'linear_penalty_r2' and
    'ours_penalty_coverage'.
    """
    train_rows, train_inputs, train_goals = _load(train_csv)
    test_rows, test_inputs, test_goals = _load(test_csv)
    n = len(test_rows)
    if n == 0 or not test_goals or not train_goals:
        return _zero()

    results = _zero()

    # Shortlist race.
    truth = _true_scores(test_rows, test_goals)
    k = max(1, min(n, int(SHORTLIST_FRAC * n)))

    def _neg_truth(i):
        t = truth[i]
        return math.inf if t is None else -t

    oracle_order = sorted(range(n), key=lambda i: (_neg_truth(i), i))
    top = set(oracle_order[:k])

    # Our shortlist: rank by the goal score estimated from inputs only.
    ests = [guess(train_csv, row) for row in test_rows]
    ranges = _train_ranges(train_rows, train_goals)
    ours_key = []
    for est in ests:
        s = _est_score(est, train_goals, ranges)
        ours_key.append(math.inf if s is None else -s)
    ours_order = sorted(range(n), key=lambda i: (ours_key[i], i))
    ours_pick = ours_order[:k]

    # Rival: top overall rating (OVR), or file order if there is no OVR column.
    if any(c == RATING_COLUMN for c in test_inputs):
        rating_key = []
        for r in test_rows:
            v = _finite(r.get(RATING_COLUMN))
            rating_key.append(math.inf if v is None else -v)
        rating_order = sorted(range(n), key=lambda i: (rating_key[i], i))
    else:
        rating_order = list(range(n))
    rating_pick = rating_order[:k]

    # Rival: random shortlist.
    rng = random.Random(SEED)
    random_pick = rng.sample(range(n), k)

    results["oracle"] = _mean_truth(oracle_order[:k], truth)
    results["ours"] = _mean_truth(ours_pick, truth)
    results["top_overall_rating"] = _mean_truth(rating_pick, truth)
    results["random_shortlist"] = _mean_truth(random_pick, truth)
    results["ours_precision"] = _precision(ours_pick, top, k)
    results["top_overall_rating_precision"] = _precision(rating_pick, top, k)
    results["random_precision"] = _precision(random_pick, top, k)

    # Penalty race.
    train_names = [c for c, _ in train_goals]
    test_names = {c for c, _ in test_goals}
    if PENALTY_GOAL in train_names and PENALTY_GOAL in test_names:
        pen = PENALTY_GOAL
    else:
        pen = next((c for c in train_names if c in test_names), None)

    if pen is not None:
        idx = [i for i, r in enumerate(test_rows) if _finite(r.get(pen)) is not None]
        if idx:
            ys = [_finite(test_rows[i].get(pen)) for i in idx]
            ours_pred = [ests[i][pen][0] for i in idx]
            lin_all = _linear_predictions(train_rows, train_inputs, test_rows, pen)
            lin_pred = [lin_all[i] for i in idx]

            covered = 0
            judged = 0
            for i, y in zip(idx, ys):
                _, low, high = ests[i][pen]
                if math.isfinite(low) and math.isfinite(high):
                    judged += 1
                    if low <= y <= high:
                        covered += 1

            results["ours_penalty_r2"] = _r2(ys, ours_pred)
            results["linear_penalty_r2"] = _r2(ys, lin_pred)
            results["ours_penalty_coverage"] = covered / judged if judged else 0.0

    return {k: float(v) for k, v in results.items()}

