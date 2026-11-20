"""Step 5: blame. Which input columns (e.g. optimisation passes) are most to blame
for bad rows, i.e. rows that do poorly on the goals.

Method: each row gets a badness score (0 for the best value on every goal, larger
for worse values, averaged over goals after normalising each goal to [0, 1]). For
each input column and each of its values, we compare the badness of rows having
that value with the badness of all other rows, using a Welch-style t statistic.
A column's blame is the largest positive t over its values, so columns whose
values sharply separate bad rows from good rows rank first.
"""
import math

from step1 import _load

MIN_GROUP = 3   # smallest number of rows a value needs on each side of the comparison


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _column_keys(rows, inputs):
    """Per input column, one key per row: a float for numeric columns, else a string."""
    keys = {}
    for c in inputs:
        raw = [r.get(c) for r in rows]
        try:
            keys[c] = [float(v) for v in raw]
        except (TypeError, ValueError):
            keys[c] = [_text(v) for v in raw]
    return keys


def _badness(rows, goals):
    """Per row: 0 if best on every goal, larger when worse. None if a goal cell is unusable."""
    vals = []
    for r in rows:
        try:
            vals.append([sign * float(r[c]) for c, sign in goals])
        except (KeyError, TypeError, ValueError):
            vals.append(None)

    ok = [v for v in vals if v is not None]
    if not ok:
        return [None] * len(rows)

    g = len(goals)
    lo = [min(v[j] for v in ok) for j in range(g)]
    hi = [max(v[j] for v in ok) for j in range(g)]

    out = []
    for v in vals:
        if v is None:
            out.append(None)
            continue
        b = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            if span > 0:
                # oriented values: higher is better, so hi - v measures how far from the best
                b += (hi[j] - v[j]) / span
        out.append(b)
    return out


def _var(n, s, ss):
    """Sample variance from count, sum and sum of squares."""
    if n < 2:
        return 0.0
    return max(0.0, (ss - s * s / n) / (n - 1))


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    rows, inputs, goals = _load(train_csv)
    if not inputs:
        return []
    if not goals or not rows:
        return list(inputs)

    bad = _badness(rows, goals)
    idx = [i for i, b in enumerate(bad) if b is not None]
    m = len(idx)
    if m < 2 * MIN_GROUP:
        return list(inputs)

    b_vals = [bad[i] for i in idx]
    total_s = sum(b_vals)
    total_ss = sum(x * x for x in b_vals)

    keys = _column_keys(rows, inputs)
    scored = []
    for pos, c in enumerate(inputs):
        col = keys[c]

        # per-value group statistics: [count, sum, sum of squares]
        stats = {}
        for k, i in enumerate(idx):
            v = col[i]
            st = stats.get(v)
            if st is None:
                st = stats[v] = [0, 0.0, 0.0]
            st[0] += 1
            st[1] += b_vals[k]
            st[2] += b_vals[k] * b_vals[k]

        best = 0.0
        for v, (nm, sm, ssm) in stats.items():
            nr = m - nm
            if nm < MIN_GROUP or nr < MIN_GROUP:
                continue
            sr = total_s - sm
            ssr = total_ss - ssm
            effect = sm / nm - sr / nr
            if effect <= 0:
                continue  # value is not associated with worse rows
            se = math.sqrt(_var(nm, sm, ssm) / nm + _var(nr, sr, ssr) / nr)
            t = effect / (se + 1e-9)
            if t > best:
                best = t

        scored.append((-best, pos, c))

    scored.sort()
    return [c for _, _, c in scored]

