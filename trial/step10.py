import csv
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _column_kind, _goal_matrix, _load
from step2 import _build_encoders, _encode, guess
from step5 import _badness
from step7 import _encode_inputs, choose


_BUDGET = 30          # labelled test builds allowed for every tuner
_GUESS_SAMPLE = 25    # test builds used to compare the two guessers (guess() is slow)
_SEED = 0


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


def _surrogate_tuner(X, badness, budget):
    """Standard-library-style rival: sequential greedy random-forest surrogate.

    Starts from a few random labels, then repeatedly labels the unlabelled
    build with the best predicted badness. Returns the best labelled index.
    """
    n = len(badness)
    rng = np.random.default_rng(_SEED)
    init = rng.choice(n, size=min(5, budget), replace=False).tolist()
    labeled = {int(i): float(badness[i]) for i in init}

    while len(labeled) < budget:
        L = sorted(labeled)
        U = [i for i in range(n) if i not in labeled]
        if not U:
            break
        y = np.array([labeled[i] for i in L], dtype=float)
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=_SEED,
            n_jobs=1,
        ).fit(X[L], y)
        pred = model.predict(X[U])
        pick = U[int(np.argmin(pred))]
        labeled[pick] = float(badness[pick])

    L = sorted(labeled)
    return int(L[int(np.argmin([labeled[i] for i in L]))])


def race(train_csv, test_csv) -> dict[str, float]:
    """Compare the earlier skills against standard rivals on test_csv.

    Tuner scores: each method labels at most _BUDGET test builds and picks one.
    The score is 1 minus the chosen build's mean normalised rank over all goals
    across the whole test set (1.0 = best possible, 0.0 = worst). Higher is better.
    Keys: "ours" (step7.choose), "random_search", "sklearn_surrogate".

    Guesser scores: negative mean absolute error of the predicted goal value
    (Performance- if present, else the first shared goal) on a fixed sample of
    test builds. Higher (closer to 0) is better.
    Keys: "ours_guess" (step2.guess), "off_the_shelf_guess" (default
    sklearn RandomForestRegressor on the same encoded inputs).
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

    # ---- tuners
    def label_fn(i):
        return {g: rows[i][g] for g in goals}

    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        _blind_copy(test_csv, blind, goals)
        pick = choose(blind, label_fn, budget=budget)
    scores["ours"] = 1.0 - float(badness[pick])

    rng = np.random.default_rng(_SEED)
    idx = rng.choice(n, size=budget, replace=False)
    pick = int(idx[int(np.argmin(badness[idx]))])
    scores["random_search"] = 1.0 - float(badness[pick])

    X_test = _encode_inputs(header, rows)
    pick = _surrogate_tuner(X_test, badness, budget)
    scores["sklearn_surrogate"] = 1.0 - float(badness[pick])

    # ---- guessers
    train_header, train_rows = _load(train_csv)
    train_goals = [c for c in train_header if _column_kind(c) in ("max", "min")]
    common = [g for g in goals if g in train_goals]
    if not common or not train_rows:
        raise ValueError("train_csv and test_csv share no goal columns or rows")
    target = "Performance-" if "Performance-" in common else common[0]

    sub = rng.choice(n, size=min(_GUESS_SAMPLE, n), replace=False)
    y_true = np.array([float(rows[i][target]) for i in sub], dtype=float)

    ours_pred = np.array(
        [float(guess(train_csv, dict(rows[i]))[target][0]) for i in sub],
        dtype=float,
    )

    train_inputs = [c for c in train_header if _column_kind(c) == "input"]
    ytr = np.array([float(r[target]) for r in train_rows], dtype=float)
    if train_inputs:
        values_by_col = {c: [r.get(c) for r in train_rows] for c in train_inputs}
        encoders = _build_encoders(values_by_col)
        Xtr = np.zeros((len(train_rows), len(train_inputs)), dtype=float)
        for j, c in enumerate(train_inputs):
            Xtr[:, j] = [_encode(v, encoders[c]) for v in values_by_col[c]]
        Xte = np.array(
            [[_encode(rows[i].get(c), encoders[c]) for c in train_inputs] for i in sub],
            dtype=float,
        ).reshape(len(sub), len(train_inputs))
        model = RandomForestRegressor(random_state=_SEED, n_jobs=1).fit(Xtr, ytr)
        off_pred = model.predict(Xte)
    else:
        off_pred = np.full(len(sub), float(np.median(ytr)), dtype=float)

    scores["ours_guess"] = -float(np.mean(np.abs(ours_pred - y_true)))
    scores["off_the_shelf_guess"] = -float(np.mean(np.abs(off_pred - y_true)))

    return scores

