"""step5.py: blame. Rank the input columns by how strongly they are tied to
bad rows. A bad row is one whose overall goal score falls in the worst quarter.

Reading, column splitting, number parsing, quantiles and the signed goal
vector come from step1, step2 and step4. Only the blame score is new here.
"""
from __future__ import annotations

import math

from step1 import _columns, _read
from step2 import _quantile
from step4 import _objectives

BAD_QUANTILE = 0.25     # rows at or below this overall score quantile are "bad"
MAX_LEVELS = 8          # numeric inputs with more distinct values are binned
BINS = 4                # quantile bins used for such numeric inputs
SHRINK = 10.0           # pseudo-count: small groups get less weight


def _scores(obj: list[list[float]], m: int) -> list[float]:
    """Mean min-max-normalised goal score per row in [0, 1], higher is better.
    Non-finite objectives count as the worst value, as in step1.best."""
    n = len(obj)
    norm = [[0.0] * m for _ in range(n)]
    for j in range(m):
        col = [o[j] for o in obj]
        fin = [x for x in col if math.isfinite(x)]
        if not fin:
            continue
        lo, hi = min(fin), max(fin)
        span = hi - lo
        for i, x in enumerate(col):
            if math.isfinite(x):
                norm[i][j] = (x - lo) / span if span > 0 else 1.0
    return [sum(v) / m for v in norm]


def _labels(rows: list[dict], c: str) -> list:
    """Group label per row for input column c. Numeric columns with few distinct
    finite values keep their values; other numeric columns are binned by
    quantile; everything else is grouped by its string form."""
    vals = [r.get(c) for r in rows]
    nums = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
    if vals and len(nums) == len(vals):
        if len(set(nums)) <= MAX_LEVELS:
            return vals
        cuts = [_quantile(nums, q / BINS) for q in range(1, BINS)]
        return [("bin", sum(1 for t in cuts if v > t)) for v in vals]
    return [str(v) for v in vals]


def _column_blame(rows: list[dict], c: str, bad: list[bool], p0: float) -> float:
    """Largest shrunken excess of the bad-row rate over the overall rate, taken
    over the groups of column c. Zero when no group is more bad than average."""
    labels = _labels(rows, c)
    n_g: dict = {}
    b_g: dict = {}
    for lab, is_bad in zip(labels, bad):
        n_g[lab] = n_g.get(lab, 0) + 1
        if is_bad:
            b_g[lab] = b_g.get(lab, 0) + 1
    best = 0.0
    for lab, ng in n_g.items():
        if ng < 2:
            continue
        rate = b_g.get(lab, 0) / ng
        score = (rate - p0) * ng / (ng + SHRINK)
        if score > best:
            best = score
    return best


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Each row gets a mean normalised goal score (higher is better), and the rows
    at or below the BAD_QUANTILE of that score are bad. An input column is blamed
    in proportion to how much more often bad rows fall in its most troublesome
    value or bin than the overall bad rate. Ties keep the file's column order.
    Every input column is returned.
    """
    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    if not inputs:
        return []
    if n == 0 or not goals:
        return list(inputs)

    obj = [_objectives(r, gmax, gmin) for r in rows]
    score = _scores(obj, len(goals))
    cut = _quantile(score, BAD_QUANTILE)
    bad = [s <= cut for s in score]
    p0 = sum(bad) / n

    blames = [_column_blame(rows, c, bad, p0) for c in inputs]
    order = sorted(range(len(inputs)), key=lambda i: (-blames[i], i))
    return [inputs[i] for i in order]

