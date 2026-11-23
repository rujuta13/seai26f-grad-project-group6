"""step25.py: name the input columns to blame for two customer questions.

Question 1, "why do some customers never respond to campaigns?": a customer is the
group of interest when ACCEPTEDCMPOVERALL is 0, meaning they never accepted any
campaign offer. The campaign-acceptance columns are left out of the ranking for this
question, since they describe the same behaviour and would trivially rank first.

Question 2, "what do the customers who complain have in common?": a customer is the
group of interest when COMPLAIN- is positive.

Each question gets its own ranking of input columns by mutual information with its
label (step5's method). The final order averages the two rankings, so columns that
matter for both questions come first. Columns excluded from a question rank last
within that question. Loading, classification, encoding and the discreteness rule
come from step1, step11 and step15.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from step1 import _classify, _input_matrix, _load
from step11 import _trait_inputs
from step15 import _DISCRETE_MAX, _SEED

_CAMPAIGN_COLS = (
    "ACCEPTEDCMP1",
    "ACCEPTEDCMP2",
    "ACCEPTEDCMP3",
    "ACCEPTEDCMP4",
    "ACCEPTEDCMP5",
    "ACCEPTEDCMPOVERALL",
)


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for the two customer questions first.

    The order averages, over the two questions, each column's rank by mutual
    information with the group of interest. Ties are broken by header order.
    """
    header, rows = _load(train_csv)
    _, _, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)
    names = [header[i] for i in inputs]
    n = len(rows)
    m = len(inputs)
    if m == 0 or n < 2:
        return names

    X = _input_matrix(rows, inputs)
    discrete = np.array(
        [len(np.unique(X[:, j])) <= _DISCRETE_MAX for j in range(m)],
        dtype=bool,
    )

    # Each question: (boolean label for the group of interest, names to leave out).
    questions = []
    if "ACCEPTEDCMPOVERALL" in header:
        col = header.index("ACCEPTEDCMPOVERALL")
        accepted = np.array([float(r[col]) for r in rows], dtype=float)
        questions.append((accepted == 0, set(_CAMPAIGN_COLS)))
    if "COMPLAIN-" in header:
        col = header.index("COMPLAIN-")
        complained = np.array([float(r[col]) for r in rows], dtype=float)
        questions.append((complained > 0, set()))

    total_pos = np.zeros(m, dtype=float)
    used = 0
    for y, excluded in questions:
        if not y.any() or y.all():
            continue
        eligible = [j for j in range(m) if names[j] not in excluded]
        if not eligible:
            continue
        mi = mutual_info_classif(
            X[:, eligible],
            y.astype(int),
            discrete_features=discrete[eligible],
            random_state=_SEED,
        )
        order = sorted(range(len(eligible)), key=lambda k: (-float(mi[k]), eligible[k]))
        pos = np.full(m, float(len(eligible)), dtype=float)   # excluded columns rank last
        for rank, k in enumerate(order):
            pos[eligible[k]] = float(rank)
        total_pos += pos
        used += 1

    if used == 0:
        # Neither question can be tested on this file, so no column can be blamed.
        return names

    mean_pos = total_pos / used
    order = sorted(range(m), key=lambda j: (float(mean_pos[j]), j))
    return [names[j] for j in order]

