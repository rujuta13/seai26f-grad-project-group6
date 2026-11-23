"""step19.py: flag player records that look like data-entry mistakes, or are unlike
the training data.

odd() is step9.odd with the identifier columns (num, Name, url) left out of the
input-novelty check, as step11, step12 and step14 do. Identifiers are row labels,
not player traits. Left in, a new player's Name is an unseen category and scores
inf, and a number column is pulled toward whichever players happen to be listed
nearby. Goal-glitch scoring and the typical-distance reference are unchanged.

Helpers come from step1 (loading, classification), step4 (input encoding and the
leave-one-out reference distance), step9 (robust goal scaling, unusable-input and
missing-goal checks, neighbour count) and step11 (identifier filter).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _load
from step4 import _encode_row, _encode_train, _loo_nearest
from step9 import _K, _goal_values, _input_unusable, _robust_scale
from step11 import _trait_inputs


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more suspicious.

    A score of inf marks a row with a missing or non-finite goal, an unparseable or
    non-finite numeric trait, or a categorical trait value never seen in train_csv.
    Identifier columns are not used for novelty.
    """
    if not rows:
        return []

    header, train_rows = _load(train_csv)
    maximise, minimise, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)
    goal_cols = minimise + maximise
    goal_names = [header[c] for c in goal_cols]
    m = len(goal_cols)
    n = len(train_rows)

    if n == 0:
        return [math.inf] * len(rows)

    # Training goals in raw units, with a robust spread per goal.
    Y = np.array([[float(r[c]) for c in goal_cols] for r in train_rows],
                 dtype=float).reshape(n, m)
    gscale = [_robust_scale(Y[:, j]) for j in range(m)]

    # Training traits: encoded, standardised, and the typical leave-one-out distance.
    X, specs = _encode_train(train_rows, inputs)
    if inputs:
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
    else:
        scaler = None
        Z = np.zeros((n, 0), dtype=float)
    ref = np.asarray(_loo_nearest(Z), dtype=float)
    typical = float(np.median(ref)) if n >= 2 else 1.0
    if not (0.0 < typical < math.inf):
        typical = 1.0

    scores: list[float] = []
    for row in rows:
        g = _goal_values(row, goal_names)
        if g is None or (inputs and _input_unusable(row, header, inputs, specs)):
            scores.append(math.inf)
            continue

        # Novelty: distance to the nearest training player, in typical-spread units.
        if inputs:
            q = _encode_row(row, header, inputs, specs)
            zq = scaler.transform(q.reshape(1, -1))[0]
            dist = np.sqrt(((Z - zq) ** 2).sum(axis=1))
        else:
            dist = np.zeros(n, dtype=float)
        novelty = float(dist.min()) / typical

        # Goal glitch: each goal against the global training spread and against the
        # goals of the nearest training players.
        k = min(_K, n)
        idx = np.argsort(dist, kind="stable")[:k]
        goal_score = 0.0
        gv = np.asarray(g, dtype=float)
        for j in range(m):
            med, scale = gscale[j]
            glob = abs(gv[j] - med) / scale
            local_med = float(np.median(Y[idx, j]))
            loc = abs(gv[j] - local_med) / scale
            goal_score = max(goal_score, glob, loc)

        scores.append(float(max(novelty, goal_score)))

    return scores

