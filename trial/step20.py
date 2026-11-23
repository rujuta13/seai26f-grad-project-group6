"""step20.py: race the shortlist and the penalty guesser against standard rivals on test_csv.

Shortlist question ("does our shortlist beat a top-overall-rating list?"):
  Every method picks k = floor(5% of test rows) players, the same cap step1 uses.
  ours         players with the best mean goal rank of step12.guess predictions,
               made from traits only (goals hidden)
  top_ovr      the k players with the highest OVR
  random       k players drawn uniformly at random (fixed seed)
  Score: mean goal-rank position of the chosen players among all test players,
  scaled to [0, 1]. Lower is better; 0 means the chosen players are the best on average.

Penalty question ("does our penalty guesser beat an off-the-shelf model?"):
  ours_penalty step12.guess estimate of Penalties+ (k-nearest-neighbour median)
  rf_penalty   random forest regressor on the same trait inputs
  Score: mean absolute error on Penalties+, divided by the training spread.
  Lower is better.

Both questions use trait inputs only (identifier columns are excluded, as in
step11 and step12), so the rivals and our methods see the same information.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _classify, _goal_matrix, _load
from step4 import _encode_row, _encode_train
from step6 import _mean_rank
from step11 import _trait_inputs
from step12 import guess

_SEED = 0                  # fixed seed so the race is repeatable
_SHORTLIST_FRACTION = 0.05  # same cap as step1.best: at most 5% of players
_RF_TREES = 200
_OVR = "OVR"
_PEN = "Penalties+"


def race(train_csv, test_csv) -> dict[str, float]:
    """Scores of our shortlist and penalty guesser against standard rivals on test_csv.

    Keys: "ours", "top_ovr", "random" (shortlist methods, see module docstring) and
    "ours_penalty", "rf_penalty" (Penalties+ guessers). All scores are lower-is-better.
    """
    header, train_rows = _load(train_csv)
    test_header, test_rows = _load(test_csv)
    if test_header != header:
        raise ValueError("train_csv and test_csv must have the same header")
    if not train_rows or not test_rows:
        raise ValueError("train_csv and test_csv must both contain data rows")

    maximise, minimise, inputs = _classify(header)
    goal_cols = minimise + maximise
    if not goal_cols:
        raise ValueError("files have no goal columns")
    if _OVR not in header:
        raise ValueError(f"column {_OVR!r} is required for the top-rating rival")
    if _PEN not in header or header.index(_PEN) not in goal_cols:
        raise ValueError(f"goal column {_PEN!r} is required for the penalty guesser")

    n = len(test_rows)
    k = max(1, int(np.floor(_SHORTLIST_FRACTION * n)))
    denom = max(n - 1, 1)
    goal_names = [header[c] for c in goal_cols]

    # True goals of the test players, as a minimisation matrix, and their mean ranks.
    G = _goal_matrix(test_rows, maximise, minimise)
    mr = _mean_rank(G)

    def shortlist_score(idx):
        """Mean position of the chosen players by mean goal rank, scaled to [0, 1]."""
        return float(np.mean(mr[np.asarray(idx, dtype=int)])) / denom

    # Our guesser: step12.guess on trait inputs only, for every goal of every test player.
    pred = np.empty((n, len(goal_cols)), dtype=float)
    for i, row in enumerate(test_rows):
        est = guess(train_csv, dict(zip(header, row)))
        pred[i] = [est[name][0] for name in goal_names]

    # Predicted goals as a minimisation matrix, ranked the same way as the truth.
    sign = np.array([1.0 if c in minimise else -1.0 for c in goal_cols], dtype=float)
    ours_idx = np.argsort(_mean_rank(pred * sign), kind="stable")[:k]

    # Rival 1: top overall rating.
    ovr_col = header.index(_OVR)
    ovr = np.array([float(r[ovr_col]) for r in test_rows], dtype=float)
    top_idx = np.argsort(-ovr, kind="stable")[:k]

    # Rival 2: random players.
    rng = np.random.default_rng(_SEED)
    rand_idx = rng.choice(n, size=k, replace=False)

    scores: dict[str, float] = {
        "ours": shortlist_score(ours_idx),
        "top_ovr": shortlist_score(top_idx),
        "random": shortlist_score(rand_idx),
    }

    # Penalty guessing.
    pc = header.index(_PEN)
    j = goal_cols.index(pc)
    truth_pen = np.array([float(r[pc]) for r in test_rows], dtype=float)
    train_pen = np.array([float(r[pc]) for r in train_rows], dtype=float)
    sd = float(np.std(train_pen))
    if sd == 0.0 or not np.isfinite(sd):
        sd = 1.0

    def pen_score(pred_pen):
        return float(np.mean(np.abs(pred_pen - truth_pen))) / sd

    scores["ours_penalty"] = pen_score(pred[:, j])

    # Off-the-shelf rival: random forest on the same encoded trait inputs.
    traits = _trait_inputs(header, inputs)
    if traits:
        Xtrain, specs = _encode_train(train_rows, traits)
        F = np.array(
            [_encode_row(dict(zip(header, r)), header, traits, specs) for r in test_rows],
            dtype=float,
        )
    else:
        Xtrain = np.zeros((len(train_rows), 1), dtype=float)
        F = np.zeros((n, 1), dtype=float)

    forest = RandomForestRegressor(n_estimators=_RF_TREES, random_state=_SEED)
    forest.fit(Xtrain, train_pen)
    scores["rf_penalty"] = pen_score(forest.predict(F))

    return scores

