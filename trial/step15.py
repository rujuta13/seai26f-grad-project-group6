"""step15.py: name the input columns to blame for the two client questions.

Question 1, "why are some players weak from the spot?": a player is bad here when
Penalties+ is in the bottom quartile of the data.

Question 2, "what separates the physically strong players from the rest?": a player
is the group of interest when Strength+ is in the top quartile of the data.

Each question gives its own ranking of input columns by mutual information with its
label (step5's method). The final order averages the two rankings, so columns that
matter for both questions come first. Identifier columns (num, Name, url) are left
out, as in step11. Loading, classification and encoding come from step1 and step11.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from step1 import _classify, _input_matrix, _load
from step11 import _trait_inputs

_QUARTILE = 0.25     # bottom (or top) share of players counted as the group of interest
_DISCRETE_MAX = 20   # columns with at most this many distinct values are treated as discrete
_SEED = 0            # fixed seed so repeated calls give the same ranking

# (goal column name, use the top quartile as the label?)
#   Penalties+ bottom quartile = weak from the spot
#   Strength+  top quartile    = physically strong
_TARGETS = (
    ("Penalties+", False),
    ("Strength+", True),
)


def _label(values, want_high):
    """0/1 label for the group of interest, or None if the split is degenerate."""
    if want_high:
        thr = np.quantile(values, 1.0 - _QUARTILE)
        y = values >= thr
    else:
        thr = np.quantile(values, _QUARTILE)
        y = values <= thr
    if not y.any() or y.all():
        return None
    return y.astype(int)


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for the two client questions first.

    The order averages, over the two questions, each column's rank by mutual
    information with the group of interest. Ties are broken by header order.
    """
    header, rows = _load(train_csv)
    _, _, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)
    names = [header[i] for i in inputs]
    n = len(rows)
    if not inputs or n < 2:
        return names

    X = _input_matrix(rows, inputs)
    discrete = np.array(
        [len(np.unique(X[:, j])) <= _DISCRETE_MAX for j in range(X.shape[1])],
        dtype=bool,
    )
    m = len(inputs)

    rank_sum = np.zeros(m, dtype=float)
    used = 0
    for goal_name, want_high in _TARGETS:
        if goal_name not in header:
            continue
        col = header.index(goal_name)
        values = np.array([float(r[col]) for r in rows], dtype=float)
        y = _label(values, want_high)
        if y is None:
            continue

        mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=_SEED)
        order = sorted(range(m), key=lambda j: (-float(mi[j]), j))
        pos = np.empty(m, dtype=float)
        for k, j in enumerate(order):
            pos[j] = k
        rank_sum += pos
        used += 1

    if used == 0:
        # Neither question can be tested on this file, so no column can be blamed.
        return names

    mean_rank = rank_sum / used
    order = sorted(range(m), key=lambda j: (float(mean_rank[j]), j))
    return [names[j] for j in order]

