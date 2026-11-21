"""Step 5: which input columns are to blame for bad rows.

blame() ranks every input (flag) column by how differently its values are
distributed among bad rows and good rows. Bad rows are the worst quarter by
the normalised goal sum, where '+' goals are negated so that higher always
means worse. A column's score is the total variation distance between the
value distributions of the bad and good rows: 0 means the column does not
separate them, 1 means the two groups never share a value. Numeric columns
with many distinct values are bucketed into quantile bins first.
"""

import bisect
import math

from step1 import _load, _split_columns

BAD_FRACTION = 0.25
MAX_BINS = 8


def _as_float(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _badness(rows, maximise, minimise):
    """Per row: normalised sum of minimisation goals (higher = worse), or None."""
    cols = [(c, -1.0) for c in maximise] + [(c, 1.0) for c in minimise]
    if not cols:
        return [None] * len(rows)

    raw = []
    for r in rows:
        vals = []
        for c, sign in cols:
            v = _as_float(r.get(c))
            if v is None:
                vals = None
                break
            vals.append(sign * v)
        raw.append(vals)

    usable = [v for v in raw if v is not None]
    if not usable:
        return [None] * len(rows)

    m = len(cols)
    lo = [min(v[j] for v in usable) for j in range(m)]
    hi = [max(v[j] for v in usable) for j in range(m)]

    out = []
    for vals in raw:
        if vals is None:
            out.append(None)
            continue
        total = 0.0
        for j in range(m):
            span = hi[j] - lo[j]
            total += (vals[j] - lo[j]) / span if span > 0 else 0.0
        out.append(total)
    return out


def _column_labels(rows, column):
    """One group label per row for a column: its value, or a quantile bin if numeric and wide."""
    vals = [r.get(column) for r in rows]
    present = [v for v in vals if v is not None]
    if present and all(isinstance(v, float) for v in present):
        distinct = sorted(set(present))
        if len(distinct) > MAX_BINS:
            edges = [
                distinct[(i * len(distinct)) // MAX_BINS]
                for i in range(1, MAX_BINS)
            ]
            return [
                "?" if v is None else bisect.bisect_right(edges, v)
                for v in vals
            ]
        return ["?" if v is None else v for v in vals]
    return ["?" if v is None else str(v) for v in vals]


def _tvd(labels, bad_idx, good_idx):
    """Total variation distance between label distributions of two index groups."""
    bad_counts = {}
    for i in bad_idx:
        bad_counts[labels[i]] = bad_counts.get(labels[i], 0) + 1
    good_counts = {}
    for i in good_idx:
        good_counts[labels[i]] = good_counts.get(labels[i], 0) + 1

    nb = len(bad_idx)
    ng = len(good_idx)
    keys = set(bad_counts) | set(good_counts)
    return 0.5 * sum(
        abs(bad_counts.get(k, 0) / nb - good_counts.get(k, 0) / ng) for k in keys
    )


def blame(train_csv) -> list[str]:
    """Return input column names, most to blame for bad rows (bad on the goals) first.

    Bad rows are the worst BAD_FRACTION of rows by normalised goal sum; the
    remaining rows are treated as good. Each input column is scored by how
    differently its values are spread across bad and good rows. Ties keep
    file order.
    """
    columns, rows = _load(train_csv)
    maximise, minimise, inputs = _split_columns(columns)
    if not inputs:
        return []

    bad_scores = _badness(rows, maximise, minimise)
    valid = [i for i, b in enumerate(bad_scores) if b is not None]
    if len(valid) < 2:
        return list(inputs)

    ranked = sorted(valid, key=lambda i: (-bad_scores[i], i))
    k = max(1, int(len(valid) * BAD_FRACTION))
    if k >= len(valid):
        k = len(valid) - 1
    bad_idx = ranked[:k]
    good_idx = ranked[k:]

    scores = []
    for c in inputs:
        labels = _column_labels(rows, c)
        scores.append(_tvd(labels, bad_idx, good_idx))

    order = sorted(range(len(inputs)), key=lambda j: (-scores[j], j))
    return [inputs[j] for j in order]

