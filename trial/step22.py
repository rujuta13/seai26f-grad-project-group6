"""Step 22: guess. Customer purchase channels and campaign response.

Serves the client's requests:
  - "Guess how much a new customer will buy online versus in store."
  - "Guess whether this customer will answer our next campaign."

Builds on step12.guess: the estimate is the median of the nearest training
customers, and low..high is their 5th..95th percentile range, blended toward
whole-training statistics for customers unlike anything in training.

Campaign response is a 0/1 goal. A median of 0/1 neighbours is too coarse to
say how likely a customer is to respond, so for 0/1 goals the estimate is the
neighbour mean, read as a probability, and the range is widened to contain
it. Blending toward the training mean for unfamiliar customers follows the
same rule as step2. Count goals (purchases per channel) keep step12's median.
"""

import math

import step2
import step12


def _is_binary(values):
    """True if the non-empty list holds only 0.0 and 1.0."""
    return bool(values) and set(values) <= {0.0, 1.0}


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a new customer `row` (goal columns already removed).

    Returns {goal: (estimate, low, high)}, where low..high is the range the true
    value is fairly sure to fall in. For 0/1 goals (such as campaign response)
    the estimate is the likelihood of a 1, in [0, 1].
    """
    result = dict(step12.guess(train_csv, row))

    cols, rows, goal_values, goal_names = step2._load(train_csv)
    if not rows:
        return result

    ranked = sorted((step2._distance(cols, tr, row), i) for i, tr in enumerate(rows))
    near = ranked[:step2._K]
    mean_d = sum(d for d, _ in near) / len(near)
    far = min(1.0, mean_d / step2._FAR)

    for g in goal_names:
        vals = goal_values.get(g, {})
        if not vals or not _is_binary(list(vals.values())):
            continue

        local = []
        for _, i in ranked:
            v = vals.get(i)
            if v is not None:
                local.append(v)
                if len(local) >= step2._K:
                    break
        if not local:
            continue

        local_mean = sum(local) / len(local)
        global_mean = sum(vals.values()) / len(vals)
        w = far if len(local) >= step2._MIN_LOCAL else 1.0
        est = (1 - w) * local_mean + w * global_mean
        est = min(1.0, max(0.0, float(est)))

        _, low, high = result.get(g, (est, -math.inf, math.inf))
        low = min(float(low), est)
        high = max(float(high), est)
        result[g] = (est, low, high)

    return result

