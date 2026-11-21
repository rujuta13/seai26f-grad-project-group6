"""Step 14: flow. Keep promising scouting rows as they arrive; warn when new rows drift.

stream(): each arriving row is kept when its normalised goal total reaches the
95th percentile of the training totals (the same rule as step4.stream). Goal
normalisation uses only the training data, and missing goal values count as
worst. Each decision depends only on train_csv and the row itself.

drift(): every _STRIDE rows, the most recent _WINDOW rows are compared with the
training data, one column at a time. Numeric columns (inputs and goals) are
compared directly. Categorical columns are reduced to how typical each value is
in the training data (its training frequency), so a new category or a shift in
the category mix shows up as a shift in that number. Each comparison is a
two-sample Kolmogorov-Smirnov test with a conservative critical value. A warning
is raised once _CONFIRM consecutive checks find a difference.
"""

import math
from collections import Counter
from functools import lru_cache

import step1

_KEEP_QUANTILE = 0.95  # keep rows scoring at or above this training quantile
_WINDOW = 50           # recent rows compared against training in drift()
_STRIDE = 10           # run a drift check every _STRIDE rows
_CONFIRM = 3           # consecutive differing checks before warning
_MIN_N = 5             # fewer usable values than this in a column -> column skipped
_KS_C = 2.05           # KS constant: per-column test level about 0.0005 (many columns)


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


def _key(v):
    """Category key for a cell: missing cells share the key ''."""
    return "" if v is None else str(v)


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _score(row, ranges):
    """Sum of normalised goal values (higher is better). Missing counts as 0."""
    total = 0.0
    for name, sign, lo, span in ranges:
        v = row.get(name)
        if not _is_num(v):
            continue  # missing counts as worst
        x = v * sign
        total += (x - lo) / span if span > 0 else 1.0
    return total


@lru_cache(maxsize=4)
def _keep_model(train_csv):
    """Parse train_csv once. Returns (ranges, threshold).

    ranges: (name, sign, lo, span) per goal over the training data.
    threshold: keep cut-off on the normalised totals, or None if no goals/rows.
    """
    header, rows = step1._read(train_csv)
    _, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]

    ranges = []
    for name, sign in goals:
        vals = [r[name] * sign for r in rows if _is_num(r.get(name))]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 0.0
        ranges.append((name, sign, lo, hi - lo))
    ranges = tuple(ranges)

    if not ranges or not rows:
        return ranges, None
    totals = sorted(_score(r, ranges) for r in rows)
    return ranges, _quantile(totals, _KEEP_QUANTILE)


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or drop (False) each incoming row, in order.

    Each decision uses only the training data and the row itself: the row is
    kept when its normalised goal total reaches the training threshold.
    """
    ranges, threshold = _keep_model(train_csv)
    if threshold is None:
        return [False] * len(rows)
    return [_score(r, ranges) >= threshold for r in rows]


@lru_cache(maxsize=4)
def _profile(train_csv):
    """Parse train_csv once. Returns one entry per input and goal column.

    Each entry is (name, kind, train, freq):
      kind "num": train is the sorted finite training values, freq is None.
      kind "cat": train is the sorted training typicality of each row's value
                  (its frequency in training), freq maps value key -> frequency.
    Columns with no usable training values are left out.
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)

    cols = []
    for name in inputs + goals_max + goals_min:
        raw = [r.get(name) for r in rows]
        present = [v for v in raw
                   if v is not None and not (isinstance(v, str) and v == "")]
        if present and all(isinstance(v, float) for v in present):
            train = sorted(v for v in present if not math.isnan(v))
            if train:
                cols.append((name, "num", train, None))
            continue

        keys = [_key(v) for v in raw]
        counts = Counter(keys)
        n = len(keys)
        freq = {k: c / n for k, c in counts.items()}
        train = sorted(freq[k] for k in keys)
        if train:
            cols.append((name, "cat", train, freq))
    return cols


def _ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic of lists a and b (ties handled)."""
    a = sorted(a)
    b = sorted(b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = a[i] if a[i] <= b[j] else b[j]
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d


def _differs(col, window):
    """True if this column of the window differs from its training values."""
    name, kind, train, freq = col
    if kind == "num":
        vals = [v for v in (r.get(name) for r in window) if _is_num(v)]
    else:
        vals = [freq.get(_key(r.get(name)), 0.0) for r in window]
    if len(vals) < _MIN_N:
        return False
    n, m = len(vals), len(train)
    return _ks(vals, train) > _KS_C * math.sqrt((n + m) / (n * m))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the new results have stopped
    looking like the training results, or None if they never do.

    At each periodic check, the most recent _WINDOW rows are compared with the
    training data for every input and goal column. A warning is raised once
    _CONFIRM consecutive checks find a difference.
    """
    cols = _profile(train_csv)
    if not cols or not rows:
        return None

    streak = 0
    for i in range(_WINDOW - 1, len(rows)):
        if (i + 1) % _STRIDE:
            continue
        window = rows[i + 1 - _WINDOW: i + 1]
        differs = any(_differs(col, window) for col in cols)
        streak = streak + 1 if differs else 0
        if streak >= _CONFIRM:
            return i
    return None

