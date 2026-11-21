"""Step 25: blame. Which customer attributes go with never responding and with complaining?

Serves the client's questions:
  - "Why do some customers never respond to campaigns?"
  - "What do the customers who complain have in common?"

Method (as step15, applied to customer data):
  - Each goal is scaled so that 1 means worst. Maximised goals are reversed,
    minimised goals are not, and missing values count as worst. Campaign
    response (RESPONSE+) and complaints (COMPLAIN-) are goals, so customers who
    never respond or who complain are the bad rows on those goals.
  - Numeric inputs: positive Spearman correlation between the input and goal
    badness. Blank cells are skipped, so a numeric column with a few blanks
    still counts as numeric.
  - Text inputs: largest positive gap between one category's mean badness and
    the overall mean. Only categories with at least _MIN_GROUP rows can earn
    blame, so a single unusual customer cannot dominate.
  - Blame is summed over goals, so an input that makes any goal bad still gets
    credit. Inputs are returned most to blame first, with ties kept in header
    order.
"""

import math

import step1

_MIN_GROUP = 5  # smallest category size that can earn blame


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


def _is_blank(v):
    """True for an empty, missing or NaN cell."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    return isinstance(v, float) and math.isnan(v)


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


def _column_kind(values):
    """'num' if every non-blank cell is a number, otherwise 'cat'."""
    present = [v for v in values if not _is_blank(v)]
    if present and all(isinstance(v, float) for v in present):
        return "num"
    return "cat"


def _numeric_blame(values, bad):
    """Positive Spearman correlation between a numeric column and badness.

    Blank or NaN cells are ignored.
    """
    pairs = [(v, b) for v, b in zip(values, bad) if _is_num(v)]
    if len(pairs) < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return max(0.0, _pearson(_ranks(xs), _ranks(ys)))


def _category_blame(values, bad):
    """Largest positive gap between one category's mean badness and the overall mean.

    Blank cells form their own category. Categories smaller than _MIN_GROUP
    rows are ignored.
    """
    n = len(bad)
    if n == 0:
        return 0.0
    overall = sum(bad) / n
    sums = {}
    counts = {}
    for v, b in zip(values, bad):
        k = "" if _is_blank(v) else str(v)
        sums[k] = sums.get(k, 0.0) + b
        counts[k] = counts.get(k, 0) + 1
    best = 0.0
    for k in counts:
        if counts[k] < _MIN_GROUP:
            continue
        gap = sums[k] / counts[k] - overall
        if gap > best:
            best = gap
    return best


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Blame is summed over all goals, so an input that makes any single goal bad
    (for example never responding to campaigns, or complaining) still ranks
    high. Inputs with no measurable association come last, in header order.
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    if not rows or not inputs:
        return list(inputs)

    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not goals:
        return list(inputs)

    bads = [_goal_badness(rows, g, sign) for g, sign in goals]

    scores = {}
    for name in inputs:
        values = [r.get(name) for r in rows]
        numeric = _column_kind(values) == "num"
        total = 0.0
        for bad in bads:
            if numeric:
                total += _numeric_blame(values, bad)
            else:
                total += _category_blame(values, bad)
        scores[name] = total

    order = sorted(range(len(inputs)), key=lambda i: (-scores[inputs[i]], i))
    return [inputs[i] for i in order]

