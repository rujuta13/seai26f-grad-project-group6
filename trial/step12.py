"""Step 12: guess a player's penalties and strength from his other numbers.

guess() reports, for every goal column of train_csv, an estimate and a range
(low, high) that the true value is expected to fall in. The estimate is the
median goal value of the most similar training rows. The range is narrow when
the player's other numbers resemble training rows and wide when they do not,
so the range is the confidence statement. The client's two requests are the
'Penalties+' and 'Strength+' entries of the result. The method is the one
implemented in step2.
"""

from step2 import guess as _guess_step2


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal of train_csv.

    `row` maps input column names to values with the goal columns removed.
    low..high is the range the true value is expected to fall in: the 10th to
    90th percentile of the similar training rows, or the 5th to 95th percentile
    of the whole training set when the row is far from all of them.
    """
    result = _guess_step2(train_csv, dict(row))
    out = {}
    for goal, (estimate, low, high) in result.items():
        out[goal] = (float(estimate), float(low), float(high))
    return out

