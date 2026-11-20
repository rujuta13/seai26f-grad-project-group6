"""Step 5: which input columns are to blame for bad rows.

A row is "bad" when it is poor on the goals taken together. Each goal is
normalised over the training rows, with its orientation fixed so that higher
is better. A row's badness is 1 minus its mean normalised goal value, so 0
means best on every goal and 1 means worst on every goal. Rows missing a goal
value are left out.

For each input column, the scored rows are grouped by their value in that
column. Numeric columns with many distinct values are binned by rank. The
column's blame is the share of badness variance explained by the groups whose
mean badness is above the overall mean (eta squared, one-sided). A column
scores high when some of its values are reliably associated with bad rows.
Columns are returned most to blame first, with ties kept in file order.
"""
import math

from step1 import _load

_MAX_LEVELS = 8   # numeric inputs with more distinct values are rank-binned
_BINS = 4


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _goals(header):
    """Return [(column, sign)]: +1.0 for '+' goals, -1.0 for '-' goals."""
    out = []
    for c in header:
        if c.endswith("+"):
            out.append((c, 1.0))
        elif c.endswith("-"):
            out.append((c, -1.0))
    return out


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _scores(rows, goals):
    """Return (scored_indices, badness) for rows with all goal values finite."""
    if not goals:
        return [], []

    vecs = []
    for r in rows:
        vec = []
        for c, sg in goals:
            f = _finite(r.get(c))
            if f is None:
                vec = None
                break
            vec.append(sg * f)
        vecs.append(vec)

    idx = [i for i, v in enumerate(vecs) if v is not None]
    if not idx:
        return [], []

    g = len(goals)
    lo = [min(vecs[i][j] for i in idx) for j in range(g)]
    hi = [max(vecs[i][j] for i in idx) for j in range(g)]

    bad = []
    for i in idx:
        total = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            total += 1.0 if span == 0 else (vecs[i][j] - lo[j]) / span
        bad.append(1.0 - total / g)
    return idx, bad


def _keys(raw):
    """Group keys for one input column over the scored rows."""
    raw = [
        None if (isinstance(v, float) and not math.isfinite(v)) else v
        for v in raw
    ]
    floats = [i for i, v in enumerate(raw) if isinstance(v, float)]
    distinct = {raw[i] for i in floats}

    bins = {}
    if len(distinct) > _MAX_LEVELS:
        order = sorted(floats, key=lambda i: (raw[i], i))
        m = len(order)
        for rank, i in enumerate(order):
            bins[i] = rank * _BINS // m

    keys = []
    for i, v in enumerate(raw):
        if v is None:
            keys.append(("missing", None))
        elif isinstance(v, float):
            keys.append(("bin", bins[i]) if i in bins else ("num", v))
        else:
            keys.append(("str", v))
    return keys


def _blame_score(scores, keys):
    """One-sided eta squared: variance share of groups worse than average."""
    n = len(scores)
    if n == 0:
        return 0.0
    mu = sum(scores) / n
    sst = sum((s - mu) ** 2 for s in scores)
    if sst <= 0.0:
        return 0.0

    groups = {}
    for s, k in zip(scores, keys):
        t = groups.setdefault(k, [0, 0.0])
        t[0] += 1
        t[1] += s

    ssb = 0.0
    for cnt, tot in groups.values():
        m = tot / cnt
        if m > mu:
            ssb += cnt * (m - mu) ** 2
    return ssb / sst


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Input columns are every column that is not a goal ('+' or '-') and not
    ignored ('X'). Ties keep their order in the file.
    """
    header, rows = _load(train_csv)
    inputs = [c for c in header if not _is_goal(c) and not c.endswith("X")]
    if not inputs:
        return []

    goals = _goals(header)
    idx, bad = _scores(rows, goals)

    position = {c: p for p, c in enumerate(header)}
    score = {}
    for c in inputs:
        raw = [rows[i].get(c) for i in idx]
        score[c] = _blame_score(bad, _keys(raw))

    return sorted(inputs, key=lambda c: (-score[c], position[c]))

