"""Step 20: race. Does our shortlist beat a simple top-rating list, and does our
penalty guesser beat an off-the-shelf model?

Standard library only. Reuses:
  - step1: loading, column splitting, goal normalisation and input encoding.
  - step2: the kNN guesser (guess) and the cached training model.
  - step4: the goal bounds used to scale errors.
  - step10: the ridge regression rival (_ridge_fit, _linear_predict, RIDGE).
  - step11: the shortlist (best).

Scores (higher is better):
  - ours: mean normalised goal value of the shortlist from step 11's best()
    on test_csv (0 = worst, 1 = best over the test file).
  - top_overall: the same score for the top-rated list: as many test rows as
    the shortlist holds, taken by highest OVR (ties by row index).
  - guesser_ours: negative mean absolute error, in training spans, of step 2's
    guess() on the Penalties+ goal.
  - guesser_ridge: the same score for a ridge regression fitted on the
    training inputs for the Penalties+ goal.
"""

from step1 import _load, _columns, _input_matrix, _normalised_goals
from step2 import _model, guess
from step4 import _goal_bounds
from step10 import _ridge_fit, _linear_predict, RIDGE
from step11 import best as _shortlist


OVR = "OVR"
PENALTY_GOAL = "Penalties+"

TOP_KEYS = ("ours", "top_overall")
GUESS_KEYS = ("guesser_ours", "guesser_ridge")


def _mean(values, idx):
    """Mean of values at the indices idx; 0.0 when idx is empty."""
    return sum(values[i] for i in idx) / len(idx) if idx else 0.0


def _top_overall(rows, count):
    """Row indices of the `count` rows with the highest OVR (ties by index)."""
    order = sorted(range(len(rows)), key=lambda i: (-float(rows[i][OVR]), i))
    return order[:count]


def _guesser_scores(train_csv, rows, inputs):
    """Negative mean absolute error (in training spans) on the Penalties+ goal,
    for step 2's guess() and for the ridge rival."""
    zeros = {"guesser_ours": 0.0, "guesser_ridge": 0.0}
    model = _model(train_csv)
    goal_names = [name for name, _ in model["goals"]]
    if PENALTY_GOAL not in goal_names or PENALTY_GOAL not in rows[0]:
        return zeros

    g = goal_names.index(PENALTY_GOAL)
    # Span of this one goal. step10's _guesser_scores indexes spans by the
    # position within the chosen names, so it is not reused for a single goal.
    lo, hi = _goal_bounds(model)[g]
    span = hi - lo if hi - lo > 0 else 1.0
    n = len(rows)

    # Our guesser (step 2). guess() reads only the input columns of the probe.
    err = 0.0
    for r in rows:
        est = guess(train_csv, {c: r[c] for c in inputs})
        err += abs(est[PENALTY_GOAL][0] - r[PENALTY_GOAL]) / span
    ours = -err / n

    # Ridge rival, fitted on the training rows only. Train and test are encoded
    # together so that training codes match model["points"], as in step 2.
    train_rows = model["rows"]
    k = len(train_rows)
    feats = _input_matrix(train_rows + rows, inputs)
    X = [[1.0] + x for x in feats[:k]]
    Y = [[t[g]] for t in model["targets"]]
    W = _ridge_fit(X, Y, RIDGE)

    err = 0.0
    for x, r in zip(feats[k:], rows):
        pred = _linear_predict(W, [1.0] + x)[0]
        err += abs(pred - r[PENALTY_GOAL]) / span
    ridge = -err / n

    return {"guesser_ours": ours, "guesser_ridge": ridge}


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores for our shortlist and penalty guesser and their rivals on test_csv.

    Keys: "ours" (step 11 shortlist), "top_overall" (top OVR list of the same
    size), "guesser_ours" (step 2 guess on Penalties+), and "guesser_ridge"
    (ridge regression on Penalties+). Higher is better for every key.
    """
    header, rows = _load(test_csv)
    inputs, goals = _columns(header)
    keys = TOP_KEYS + GUESS_KEYS
    n = len(rows)
    if n == 0 or not goals:
        return {k: 0.0 for k in keys}

    # True quality of every test build, normalised over the test file.
    truth = [sum(p) / len(p) for p in _normalised_goals(rows, goals)]

    results = {}

    shortlist = _shortlist(test_csv)
    results["ours"] = _mean(truth, shortlist)

    if OVR in header:
        top = _top_overall(rows, len(shortlist))
    else:
        top = []
    results["top_overall"] = _mean(truth, top)

    results.update(_guesser_scores(train_csv, rows, inputs))

    return {k: float(results[k]) for k in keys}

