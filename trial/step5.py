"""Step 5: blame. Which input columns (flags/passes) are to blame for bad rows?

Method: each goal is scaled so that 1 means worst. Maximised goals are
reversed, minimised goals are not, and missing values count as worst. For each
goal, an input column's blame is how strongly its value tracks that goal's
badness. For numeric columns this is the positive part of a Spearman rank
correlation. For categorical columns it is the largest gap between one
category's mean badness and the overall mean. Blame is summed over goals, so an
input that makes any single goal bad (for example CPU use on builds that are
otherwise fast) still gets credit. Inputs are returned most to blame first,
with ties kept in header order.
"""

import math

from step1 import _read, _classify


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


def _goal_badness(rows, name, sign):
    """Per-row badness of one goal in [0, 1]; 1 is worst. Missing counts as 1."""
    vals = [r.get(name) for r in rows]
    scaled = [v * sign if _is_num(v) else None for v in vals]
    present = [x for x in scaled if x is not None]
    if not present:
        return [1.0] * len(rows)
    lo, hi = min(present), max(present)
    span = hi - lo
    out = []
    for x in scaled:
        if x is None:
            out.append(1.0)
        elif span > 0:
            out.append((hi - x) / span)
        else:
            out.append(0.0)
    return out


def _ranks(xs):
    """Average ranks (1-based) of xs, with ties sharing their mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _numeric_blame(values, bad):
    """Positive Spearman correlation between a numeric column and badness."""
    pairs = [(v, b) for v, b in zip(values, bad) if _is_num(v)]
    if len(pairs) < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return max(0.0, _pearson(_ranks(xs), _ranks(ys)))


def _category_blame(values, bad):
    """Largest positive gap between one category's mean badness and the overall mean."""
    n = len(bad)
    if n == 0:
        return 0.0
    overall = sum(bad) / n
    sums = {}
    counts = {}
    for v, b in zip(values, bad):
        k = str(v)
        sums[k] = sums.get(k, 0.0) + b
        counts[k] = counts.get(k, 0) + 1
    best = 0.0
    for k in counts:
        gap = sums[k] / counts[k] - overall
        if gap > best:
            best = gap
    return best


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Blame is summed over all goals, so an input that makes any single goal bad
    still ranks high. Inputs with no measurable association come last, in
    header order.
    """
    header, rows = _read(train_csv)
    inputs, goals_max, goals_min = _classify(header)
    if not rows or not inputs:
        return list(inputs)

    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not goals:
        return list(inputs)

    bads = [_goal_badness(rows, g, sign) for g, sign in goals]

    scores = {}
    for name in inputs:
        values = [r.get(name) for r in rows]
        numeric = all(isinstance(v, float) for v in values)
        total = 0.0
        for bad in bads:
            if numeric:
                total += _numeric_blame(values, bad)
            else:
                total += _category_blame(values, bad)
        scores[name] = total

    order = sorted(range(len(inputs)), key=lambda i: (-scores[inputs[i]], i))
    return [inputs[i] for i in order]

