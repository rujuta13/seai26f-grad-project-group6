import numpy as np

from step19 import _BIG, _goal_outlier, _kind, _load, _missing, _novelty, _num


# Columns whose values must fall inside a fixed range, whatever the training data says.
_PLAUSIBLE = {"AGE": (0.0, 110.0)}


def _cell_scores(header, train, rows):
    """Per-row score from individual numeric input cells.

    For each input column that is fully numeric in train:
      * a non-numeric or non-finite entry in a numeric column scores very high
        (a typo such as "3O" or "n/a");
      * a negative value in a column that is never negative in train scores very high;
      * a value outside a fixed plausible range (e.g. AGE) scores very high;
      * otherwise the robust z-score (median/MAD on train) is used. Columns whose
        MAD is zero (for example 0/1 flags) get no z-score, since any deviation
        would look extreme without being unusual.
    Missing cells are not treated as errors here.
    """
    out = np.zeros(len(rows), dtype=float)
    for c in header:
        if _kind(c) != "input":
            continue
        nums = [_num(r.get(c)) for r in train if not _missing(r.get(c))]
        if not nums or any(x is None for x in nums):
            continue
        ys = np.asarray(nums, dtype=float)
        med = float(np.median(ys))
        mad = 1.4826 * float(np.median(np.abs(ys - med)))
        nonneg = float(ys.min()) >= 0.0
        bounds = _PLAUSIBLE.get(c.upper())

        for i, r in enumerate(rows):
            raw = r.get(c)
            if _missing(raw):
                continue
            x = _num(raw)
            if x is None:
                z = _BIG
            else:
                z = 0.0
                if mad > 0:
                    z = min(abs(x - med) / mad, _BIG)
                if (nonneg and x < 0) or (bounds is not None and not (bounds[0] <= x <= bounds[1])):
                    z = _BIG
            out[i] = max(out[i], z)
    return out


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included); higher means more
    likely a glitch, a typo, or unlike anything in train_csv.

    The score is a log-scale sum of three signals:
      * cell checks: non-numeric or impossible values (negative counts or
        incomes, ages outside 0..110) and robust outliers within a single
        input column;
      * goal outlier: the largest robust z-score over the goal columns, with a
        missing or non-numeric goal scored very high;
      * input novelty: nearest-neighbour distance to the training inputs,
        relative to the typical leave-one-out distance within the training set.
    With no training rows there is nothing to compare against, so all scores
    are 0.0.

    These scores flag records for review. They do not show fraud: an unusual
    record may be genuine, and a typical-looking record may still be wrong.
    """
    rows = list(rows)
    if not rows:
        return []

    header, train = _load(train_csv)
    if not train:
        return [0.0] * len(rows)

    cell = _cell_scores(header, train, rows)
    goal = _goal_outlier(header, train, rows)
    novelty = _novelty(header, train, rows)

    score = np.log1p(cell) + np.log1p(goal) + np.log1p(novelty)
    score = np.where(np.isfinite(score), score, _BIG)
    return [float(s) for s in score]

