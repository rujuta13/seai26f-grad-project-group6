import csv
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _column_kind, _goal_matrix, _load
from step5 import _badness
from step7 import choose
from step12 import (
    _encode_row,
    _load as _load_raw,
    _num,
    _spec,
    _usable_inputs,
    guess as _penalty_guess,
)


_BUDGET = 30          # labelled test builds allowed for every tuner
_GUESS_SAMPLE = 25    # test builds used to compare the two penalty guessers
_SEED = 0
_RATING = "OVR"       # the "top overall rating" list is ordered by this column
_PENALTY = "Penalties+"


def _blind_copy(src_csv, dst_csv, goal_cols):
    """Write a copy of src_csv with every goal cell replaced by '?'."""
    with open(src_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        raw = list(reader)
    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in raw:
            out = dict(r)
            for g in goal_cols:
                out[g] = "?"
            writer.writerow(out)


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare the earlier skills against standard rivals on test_csv.

    Shortlist question: each method labels at most _BUDGET test builds and
    picks one. The score is 1 minus the chosen build's mean normalised rank
    over all goals across the test set (1.0 = best, 0.0 = worst). Keys:
      "ours"               step7.choose, budgeted search over all test builds
      "random_search"      best of a random sample of _BUDGET test builds
      "top_overall_rating" best of the _BUDGET test builds with the highest
                           OVR (omitted if the test data has no OVR column)

    Penalty question: negative mean absolute error of the predicted
    Penalties+ on a fixed sample of test builds (higher, closer to 0, is
    better). Keys:
      "ours_penalty_guess"          step12.guess (our penalty guesser)
      "off_the_shelf_penalty_guess" default RandomForestRegressor on the
                                    same encoded inputs
    """
    header, rows = _load(test_csv)
    n = len(rows)
    goals = [c for c in header if _column_kind(c) in ("max", "min")]
    if n == 0 or not goals:
        raise ValueError("test_csv needs rows and goal columns")

    G = _goal_matrix(header, rows)
    badness = _badness(G)
    budget = min(_BUDGET, n)
    scores: dict[str, float] = {}
    rng = np.random.default_rng(_SEED)

    # ---- shortlist question
    def label_fn(i):
        return {g: rows[i][g] for g in goals}

    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _blind_copy(test_csv, blind, goals)
        pick = choose(blind, label_fn, budget=budget)
    scores["ours"] = 1.0 - float(badness[pick])

    idx = rng.choice(n, size=budget, replace=False)
    pick = int(idx[int(np.argmin(badness[idx]))])
    scores["random_search"] = 1.0 - float(badness[pick])

    if _RATING in header:
        ovr = np.empty(n, dtype=float)
        for i, r in enumerate(rows):
            v = _num(r.get(_RATING))
            ovr[i] = -np.inf if v is None else v
        order = sorted(range(n), key=lambda i: (-ovr[i], i))
        top = order[:budget]
        pick = min(top, key=lambda i: badness[i])
        scores["top_overall_rating"] = 1.0 - float(badness[pick])

    # ---- penalty-guesser question
    raw_train_header, raw_train = _load_raw(train_csv)
    _, raw_test = _load_raw(test_csv)
    if _PENALTY not in goals or _PENALTY not in raw_train_header:
        raise ValueError(
            f"'{_PENALTY}' must be a goal column in both train_csv and test_csv"
        )

    sample = rng.choice(n, size=min(_GUESS_SAMPLE, n), replace=False)
    sub = [int(i) for i in sample if _num(raw_test[int(i)].get(_PENALTY)) is not None]
    if not sub:
        raise ValueError("test_csv has no usable Penalties+ values to compare against")
    y_true = np.array([_num(raw_test[i][_PENALTY]) for i in sub], dtype=float)

    ours_pred = np.array(
        [
            float(_penalty_guess(train_csv, dict(raw_test[i]))[_PENALTY][0])
            for i in sub
        ],
        dtype=float,
    )

    tr_rows, tr_y = [], []
    for r in raw_train:
        y = _num(r.get(_PENALTY))
        if y is not None:
            tr_rows.append(r)
            tr_y.append(y)
    if not tr_rows:
        raise ValueError("train_csv has no usable Penalties+ values")
    ytr = np.asarray(tr_y, dtype=float)

    train_inputs = _usable_inputs(raw_train_header, raw_train)
    if train_inputs:
        specs = [_spec(c, [r.get(c) for r in tr_rows]) for c in train_inputs]
        Xtr = np.vstack([_encode_row(r, specs) for r in tr_rows])
        Xte = np.vstack([_encode_row(raw_test[i], specs) for i in sub])
        model = RandomForestRegressor(random_state=_SEED, n_jobs=1).fit(Xtr, ytr)
        off_pred = model.predict(Xte)
    else:
        off_pred = np.full(len(sub), float(np.median(ytr)), dtype=float)

    scores["ours_penalty_guess"] = -float(np.mean(np.abs(ours_pred - y_true)))
    scores["off_the_shelf_penalty_guess"] = -float(np.mean(np.abs(off_pred - y_true)))

    return scores

