"""Step 22: guess a customer's online and in-store purchases, and campaign response.

guess() reports, for every goal column of train_csv (including NUMWEBPURCHASES+,
NUMSTOREPURCHASES+ and RESPONSE+), an estimate and a range (low, high) that the
true value is expected to fall in. The estimate is the median goal value of the
most similar training customers. The range is narrow when the customer resembles
training customers and wide when they do not. The method is the one implemented
in step2.
"""

from step2 import guess as _guess_step2


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Return {goal: (estimate, low, high)} for every goal of train_csv.

    `row` maps input column names to values with the goal columns removed.
    low..high is the range the true value is expected to fall in.
    """
    result = _guess_step2(train_csv, dict(row))
    out = {}
    for goal, (estimate, low, high) in result.items():
        out[goal] = (float(estimate), float(low), float(high))
    return out

