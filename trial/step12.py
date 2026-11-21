"""step12.py - client question 12: guess.

guess(train_csv, row) estimates every goal of a player (flag set) that has not
been benchmarked, and gives a range the true value should fall in.

Estimate: step2.guess, the median goal value of the K nearest training flag
sets.

Range: step2's low..high only shows how much those nearest neighbours disagree.
It does not show how far a median guess tends to be from the truth. So the
range is calibrated on the training data. For a sample of training rows, each
goal is guessed from all other rows (leave-one-out), and the 10th and 90th
percentiles of the errors (true minus guessed) give the typical error band.
For a new row, that band is added to the estimate, and the result is widened
if step2's neighbour range is wider. The interval is clipped to the range of
goal values seen in training. For training-like rows it should contain the
true value about 80% of the time. A narrow interval means a confident guess;
a wide one means the guess is shaky.

Reused from earlier steps:
  step1: SEED, _sqdist
  step2: K_NEIGHBOURS, _model, _quantile, guess (as _guess_neighbours)
"""
import heapq
import random
import statistics

from step1 import SEED, _sqdist
from step2 import K_NEIGHBOURS, _model, _quantile
from step2 import guess as _guess_neighbours

CAL_ROWS = 400
COVER_LOW = 0.10
COVER_HIGH = 0.90

_BANDS = {}  # train_csv -> per goal: (err_low, err_high, goal_min, goal_max)


def _calibrate(train_csv):
    """Typical leave-one-out error band and training range of each goal."""
    if train_csv in _BANDS:
        return _BANDS[train_csv]

    input_cols, goal_cols, means, stds, Z, goal_vals = _model(train_csv)
    n, m = len(Z), len(goal_cols)

    goal_min = [min(goal_vals[i][g] for i in range(n)) for g in range(m)]
    goal_max = [max(goal_vals[i][g] for i in range(n)) for g in range(m)]

    k = min(K_NEIGHBOURS, n - 1)
    errs = [[] for _ in range(m)]
    if k >= 1:
        rng = random.Random(SEED)
        idx = rng.sample(range(n), min(CAL_ROWS, n))
        for i in idx:
            heads = heapq.nsmallest(
                k, ((_sqdist(Z[i], Z[j]), j) for j in range(n) if j != i))
            nbrs = [j for _, j in heads]
            for g in range(m):
                est = statistics.median([goal_vals[j][g] for j in nbrs])
                errs[g].append(goal_vals[i][g] - est)

    bands = []
    for g in range(m):
        if errs[g]:
            e = sorted(errs[g])
            e_low = _quantile(e, COVER_LOW)
            e_high = _quantile(e, COVER_HIGH)
        else:
            e_low = e_high = 0.0
        bands.append((e_low, e_high, goal_min[g], goal_max[g]))

    _BANDS[train_csv] = bands
    return bands


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal of a flag set and a range the true value should
    fall in (about 80% coverage on training-like rows).

    `row` maps input column names to values (goal columns removed).
    Returns {goal: (estimate, low, high)} for every goal column.
    """
    base = _guess_neighbours(train_csv, row)
    input_cols, goal_cols, means, stds, Z, goal_vals = _model(train_csv)
    bands = _calibrate(train_csv)

    result = {}
    for g, (name, _) in enumerate(goal_cols):
        est, n_low, n_high = base[name]
        e_low, e_high, g_min, g_max = bands[g]

        low = min(est + e_low, n_low, est)
        high = max(est + e_high, n_high, est)

        # a goal cannot leave the range seen in training
        low = max(low, g_min)
        high = min(high, g_max)

        result[name] = (float(est), float(low), float(high))
    return result

