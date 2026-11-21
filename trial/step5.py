"""step5.py - client question 6: blame.

blame(train_csv) ranks the input columns (flags/passes) by how strongly they
are tied to bad rows, where a row is bad when it is poor on the goals.

- Each goal is turned into a percentile rank among the training rows (0 =
  worst, 1 = best), so goals with different units count equally. Goals come
  from step1._load, which already makes minimised goals maximisation-signed.
- A row's badness is 1 minus its mean percentile over all goals. The worst
  BAD_FRACTION of rows by badness are "bad"; the rest are "good".
- For each input column, the effect is the difference between its mean among
  bad rows and among good rows, divided by the pooled within-group standard
  deviation. This standardised mean difference makes 0/1 flags and numeric
  inputs comparable. A positive effect means the input is on, or higher, more
  often in bad rows.
- Columns are returned by effect, largest first. Ties keep file order.
"""
import math

from step1 import _load

BAD_FRACTION = 0.25


def _percentiles(values):
    """Percentile rank of each value in [0, 1] (0 = smallest). Ties share the
    mean rank of their group."""
    n = len(values)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    pct = [0.0] * n
    p = 0
    while p < n:
        q = p
        while q + 1 < n and values[order[q + 1]] == values[order[p]]:
            q += 1
        mid = (p + q) / 2
        for k in range(p, q + 1):
            pct[order[k]] = mid / (n - 1)
        p = q + 1
    return pct


def _effect(bad_vals, good_vals):
    """Standardised mean difference, bad minus good."""
    nb, ng = len(bad_vals), len(good_vals)
    mb = sum(bad_vals) / nb
    mg = sum(good_vals) / ng
    ss = (sum((x - mb) ** 2 for x in bad_vals)
          + sum((x - mg) ** 2 for x in good_vals))
    sd = math.sqrt(ss / (nb + ng))
    if sd == 0:
        return 0.0
    return (mb - mg) / sd


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    input_cols, goal_cols, X, Y = _load(train_csv)
    n = len(X)
    d = len(input_cols)
    m = len(goal_cols)
    if n == 0 or d == 0 or m == 0:
        return list(input_cols)

    # badness: 1 minus the mean percentile over all goals (1 = worst)
    pct_cols = [_percentiles([Y[i][g] for i in range(n)]) for g in range(m)]
    badness = [1.0 - sum(col[i] for col in pct_cols) / m for i in range(n)]

    k = max(1, int(BAD_FRACTION * n))
    order = sorted(range(n), key=lambda i: (-badness[i], i))
    bad_idx = sorted(order[:k])
    bad_set = set(bad_idx)
    good_idx = [i for i in range(n) if i not in bad_set]
    if not good_idx:
        return list(input_cols)

    effects = [
        _effect([X[i][t] for i in bad_idx], [X[i][t] for i in good_idx])
        for t in range(d)
    ]
    ranked = sorted(range(d), key=lambda t: (-effects[t], t))
    return [input_cols[t] for t in ranked]

