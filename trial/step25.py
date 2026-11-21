"""Step 25: blame. Which input columns are most to blame for customers who do badly on
the goals: the ones who never respond to campaigns (RESPONSE+) and the ones who
complain (COMPLAIN-).

Method: each row gets a badness score. It is 0 for the best value on every goal and
larger for worse values, averaged over goals after scaling each goal to [0, 1]. The
goals used are the client's two questions, RESPONSE+ and COMPLAIN-, when the file has
them. Otherwise all goal columns are used. For each input column and each of its
values, the badness of rows with that value is compared with the badness of all other
rows using a Welch-style t statistic. Only values associated with worse rows count. A
column's blame is the largest such t over its values, so columns whose values sharply
separate bad customers from good ones rank first.

Standard library only. Missing or non-finite cells do not crash the function: rows
with an unusable goal cell are left out of the comparison, and a missing or text input
cell is its own group.
"""
import math

from step1 import _load

MIN_GROUP = 3                          # smallest number of rows a value needs on each side
FOCUS_GOALS = ("RESPONSE+", "COMPLAIN-")  # goals behind the client's two questions


def _finite(v):
    """The cell as a finite float, or None if it is missing, non-numeric, or non-finite."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _key(v):
    """Group key for one input cell: its finite number if it has one, else its text
    ('' for a missing cell)."""
    x = _finite(v)
    if x is not None:
        return x
    return "" if v is None else str(v)


def _badness(rows, goals):
    """Per row: 0 if best on every goal, larger when worse, averaged over goals.
    None if any goal cell of the row is unusable."""
    oriented = []
    for r in rows:
        vec = []
        for c, sign in goals:
            x = _finite(r.get(c))
            if x is None:
                vec = None
                break
            vec.append(sign * x)
        oriented.append(vec)

    ok = [v for v in oriented if v is not None]
    if not ok:
        return [None] * len(rows)

    g = len(goals)
    lo = [min(v[j] for v in ok) for j in range(g)]
    hi = [max(v[j] for v in ok) for j in range(g)]

    out = []
    for v in oriented:
        if v is None:
            out.append(None)
            continue
        b = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            if span > 0:
                # oriented values: higher is better, so hi - v measures distance from the best
                b += (hi[j] - v[j]) / span
        out.append(b / g)
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

    focus = [g for g in goals if g[0] in FOCUS_GOALS]
    if focus:
        goals = focus

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

    scored = []
    for pos, c in enumerate(inputs):
        keys = [_key(rows[i].get(c)) for i in idx]

        # per-value group statistics: [count, sum, sum of squares]
        stats = {}
        for k, key in enumerate(keys):
            st = stats.get(key)
            if st is None:
                st = stats[key] = [0, 0.0, 0.0]
            st[0] += 1
            st[1] += b_vals[k]
            st[2] += b_vals[k] * b_vals[k]

        best = 0.0
        for nm, sm, ssm in stats.values():
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

