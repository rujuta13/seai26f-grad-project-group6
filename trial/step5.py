"""Step 5: blame. Which input columns are most to blame for bad rows?

Standard library only. Loading, column splitting and goal normalisation are
reused from step1.

Method: a row is bad when its mean normalised goal value (higher is better)
is in the worst quarter of the training rows. For every input column and
every value it takes in at least MIN_GROUP rows, the lift is how much more
often that value occurs among bad rows than among all rows. A column's blame
is the share-weighted sum of its positive lifts, so columns that concentrate
bad rows on common values rank first.
"""

from step1 import _load, _columns, _normalised_goals


BAD_SHARE = 0.25   # worst share of rows (by goal score) counted as bad
MIN_GROUP = 5      # values seen in fewer rows are ignored as noise


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    header, rows = _load(train_csv)
    n = len(rows)
    inputs, goals = _columns(header)
    if n == 0 or not inputs or not goals:
        return list(inputs)

    points = _normalised_goals(rows, goals)
    score = [sum(p) / len(p) for p in points]
    n_bad = max(1, int(n * BAD_SHARE))
    order = sorted(range(n), key=lambda i: (score[i], i))
    bad = set(order[:n_bad])
    base = n_bad / n

    blames = []
    for col_index, c in enumerate(inputs):
        # value -> [rows with this value, bad rows with this value]
        groups = {}
        for i, row in enumerate(rows):
            g = groups.setdefault(row[c], [0, 0])
            g[0] += 1
            if i in bad:
                g[1] += 1

        total = 0.0
        for count, bad_count in groups.values():
            if count < MIN_GROUP:
                continue
            lift = bad_count / count - base
            if lift > 0:
                total += lift * count / n
        blames.append((-total, col_index, c))

    blames.sort()
    return [c for _, _, c in blames]

