"""step9.py - client question 9: spot.

odd(train_csv, rows) gives one suspicion score per benchmark row (goals
included). Higher means more likely a glitch, a typo, or unlike anything in
train_csv. A row can be suspicious for two reasons:

  flags: its flag set lies far from every training flag set (an unusual or
         mistyped setting). The distance is the standardised nearest-training
         distance used by step4, divided by the typical leave-one-out nearest
         distance among the training rows.
  goals: a goal value disagrees with what its flag set's nearest training
         neighbours give (a glitch or typo in a measurement). The deviation
         from step2.guess's estimate is divided by a robust spread of that
         goal, measured on the training rows with leave-one-out neighbours.

Both parts are calibrated on train_csv, so a score near 1 is ordinary for
training-like data. The final score is sqrt(flag_part**2 + goal_part**2),
where goal_part is the largest goal deviation over the goals present in the row.

Reused from earlier steps:
  step1: _sqdist
  step2: _model (standardised flags, goal values in original units),
         guess (neighbour estimate of each goal), K_NEIGHBOURS
"""
import heapq
import math
import statistics

from step1 import _sqdist
from step2 import K_NEIGHBOURS, _model, guess

# train_csv path -> (flag_scale, goal_scale), computed once per training file
_CALIBRATION = {}


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row (goals included). Around 1 is typical for
    training-like data; larger means more likely a glitch, a typo, or unlike
    anything in train_csv."""
    input_cols, goal_cols, means, stds, Z, goal_vals = _model(train_csv)
    n, m = len(Z), len(goal_cols)
    if n == 0:
        raise ValueError("training file has no data rows")

    if train_csv not in _CALIBRATION:
        # Leave-one-out behaviour of the training rows gives the typical scale
        # of both the flag distance and each goal's neighbour deviation.
        flag_nn = []
        goal_dev = [[] for _ in range(m)]
        k = min(K_NEIGHBOURS, n - 1)
        if k >= 1:
            for i in range(n):
                heads = heapq.nsmallest(
                    k, ((_sqdist(Z[i], Z[j]), j) for j in range(n) if j != i))
                nbrs = [j for _, j in heads]
                flag_nn.append(math.sqrt(heads[0][0]))
                for g in range(m):
                    est = statistics.median([goal_vals[j][g] for j in nbrs])
                    goal_dev[g].append(abs(goal_vals[i][g] - est))

        positive_nn = [x for x in flag_nn if x > 0]
        flag_scale = statistics.median(positive_nn) if positive_nn else 1.0

        goal_scale = []
        for g in range(m):
            s = 0.0
            if goal_dev[g]:
                # 1.4826 * median absolute deviation: a robust sigma
                s = 1.4826 * statistics.median(goal_dev[g])
            if s <= 0 and n > 1:
                s = statistics.pstdev([goal_vals[i][g] for i in range(n)])
            goal_scale.append(s if s > 0 else 1.0)

        _CALIBRATION[train_csv] = (flag_scale, goal_scale)

    flag_scale, goal_scale = _CALIBRATION[train_csv]

    scores = []
    for row in rows:
        flags = {c: float(row[c]) for c in input_cols}
        z = [(flags[c] - means[t]) / stds[t] for t, c in enumerate(input_cols)]
        nn_sq = min(_sqdist(z, zi) for zi in Z)
        flag_part = math.sqrt(nn_sq) / flag_scale

        est = guess(train_csv, flags)
        goal_part = 0.0
        for g, (name, _) in enumerate(goal_cols):
            if name in row:
                dev = abs(float(row[name]) - est[name][0]) / goal_scale[g]
                goal_part = max(goal_part, dev)

        scores.append(math.hypot(flag_part, goal_part))
    return scores

