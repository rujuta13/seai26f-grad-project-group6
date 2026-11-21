"""step9.py: spot benchmark results that look like glitches, and flag sets unlike any seen.

Two kinds of suspicion are combined:
- Goal-side: a row whose goal values are missing, non-numeric or non-finite is
  treated as a certain glitch. Otherwise its goals are compared with the
  training goals, both by single-goal extremeness and by distance to the
  nearest training row in goal space.
- Input-side: a flag set whose settings sit far from every training flag set
  (in standardised encoded input space) is unlike anything seen.

Each distance is divided by the typical leave-one-out nearest-neighbour distance
inside train_csv, so a score of about 1 means "as isolated as a normal training
row" and larger values mean more suspicious.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _load, _encode
from step6 import _num

# Score given to rows that cannot be trusted as measurements at all.
_GLITCH = float("inf")


def _goal_array(rows, goals):
    """Raw goal values as floats, nan where a cell is missing, non-numeric or non-finite."""
    out = np.full((len(rows), len(goals)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, (name, _) in enumerate(goals):
            v = _num(r.get(name))
            if v is not None:
                out[i, j] = v
    return out


def _nn_dist(A, B):
    """Distance from each row of A to its nearest row of B (inf if B is empty)."""
    if B.shape[0] == 0:
        return np.full(A.shape[0], np.inf)
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return d.min(axis=1)


def _typical(T):
    """Median leave-one-out nearest-neighbour distance within T (1.0 if undefined)."""
    n = T.shape[0]
    if n < 2 or T.shape[1] == 0:
        return 1.0
    d = np.linalg.norm(T[:, None, :] - T[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 1.0
    med = float(np.median(nn))
    return med if med > 0 else 1.0


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more likely
    a glitch, a typo, or unlike anything in train_csv.

    A row with a missing or non-numeric goal scores infinity. Otherwise the score is
    the larger of its goal-side and input-side isolation ratios, where a ratio of 1
    means it is about as isolated as a typical training row. Single goals more than
    three standard deviations from the training mean also raise the goal-side score.
    """
    rows = list(rows)
    if not rows:
        return []

    train_rows, inputs, goals = _load(train_csv)
    n_q = len(rows)

    # Goal side.
    if goals:
        G_tr = _goal_array(train_rows, goals)
        G_q = _goal_array(rows, goals)
        missing = ~np.isfinite(G_q).all(axis=1)

        mu = np.zeros(len(goals))
        sd = np.ones(len(goals))
        for j in range(len(goals)):
            col = G_tr[:, j]
            finite = col[np.isfinite(col)]
            if finite.size:
                mu[j] = finite.mean()
                s = finite.std()
                sd[j] = s if s > 0 else 1.0

        Z_tr = np.nan_to_num((G_tr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        Z_q = np.nan_to_num((G_q - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

        if train_rows:
            ratio_goal = _nn_dist(Z_q, Z_tr) / _typical(Z_tr)
        else:
            ratio_goal = np.zeros(n_q)
        zmax = np.abs(Z_q).max(axis=1)
        goal_score = np.maximum(ratio_goal, zmax / 3.0)
    else:
        missing = np.zeros(n_q, dtype=bool)
        goal_score = np.zeros(n_q)

    # Input side: how isolated each flag set is among the training flag sets.
    if inputs and train_rows:
        query = [{c: r.get(c, "") for c in inputs} for r in rows]
        X_all = _encode(train_rows + query, inputs)
        n_tr = len(train_rows)
        X_tr, X_q = X_all[:n_tr], X_all[n_tr:]
        scaler = StandardScaler().fit(X_tr)
        T = scaler.transform(X_tr)
        Q = scaler.transform(X_q)
        input_score = _nn_dist(Q, T) / _typical(T)
    else:
        input_score = np.zeros(n_q)

    combined = np.maximum(goal_score, input_score)
    combined = np.where(np.isnan(combined), 0.0, combined)
    combined[missing] = _GLITCH
    return [float(x) for x in combined]

