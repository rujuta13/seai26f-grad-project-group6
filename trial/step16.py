"""Step 16: justify.

explain(train_csv): plain-language reasons a manager can check for why the good
players in train_csv are worth signing. Every reason is a count, median or
comparison that can be recomputed from the training csv.

why(train_csv, row): one or two sentences on why one player is rated good or
low, with reasons that can be checked on video.

Shared definitions:
  - Goal score: each goal is scaled to [0, 1] over the training players with
    complete goal values (0 = worst, 1 = best) and averaged over goals.
  - Good players: the Pareto front of the goals (no other player is at least as
    good on every goal and better on one), capped at 5% of rows, as step11.best
    returns them.
  - Rated good: no training player dominates the player on the goals. Rated low:
    at least one does.
"""
import math
from collections import Counter

from step1 import _load
from step11 import best
from step13 import same

MIN_SUPPORT = 3      # a value must appear at least this often on both sides of a comparison
TOP_SETTINGS = 5     # how many associated attribute values explain() tests and reports
TOP_PROFILE = 5      # how many numeric inputs explain() reports
MAX_CATEGORICAL = 8  # how many common categorical attributes explain() reports
MAX_LEVELS = 30      # text columns with more distinct values are identifier-like and skipped
MAJORITY = 0.6       # a common value is flagged "mixed" below this share of good players
NUMERIC_FRAC = 0.9   # a column is numeric if this share of its cells parse as numbers
ID_FRAC = 0.9        # a numeric column with more distinct values than this share of rows is an id
MIN_DIFF = 0.5       # a numeric input this many spreads from the good-player median is a reason
MAX_TEST = 1000      # permutation tests use at most this many evenly spaced players per group


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _text(v):
    if isinstance(v, str):
        return v.strip()
    return "" if v is None else str(v).strip()


def _oriented(row, goals):
    """Goal vector with higher always better, or None if any goal cell is unusable."""
    out = []
    for c, sign in goals:
        x = _finite(row.get(c))
        if x is None:
            return None
        out.append(sign * x)
    return out


def _dominates(q, p):
    """True if q is at least as good as p on every goal and better on at least one."""
    return all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p))


def _scale(vectors, g):
    lo = [min(v[j] for v in vectors) for j in range(g)]
    hi = [max(v[j] for v in vectors) for j in range(g)]
    return lo, hi


def _score(o, lo, hi):
    """Mean of the per-goal scaled values; a constant goal contributes 0, as in step1."""
    total = 0.0
    for j in range(len(o)):
        span = hi[j] - lo[j]
        if span > 0:
            total += (o[j] - lo[j]) / span
    return total / len(o)


def _median(vals):
    """Median of a non-empty list of numbers."""
    s = sorted(vals)
    m = len(s)
    mid = m // 2
    if m % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _quantile(vals, q):
    """Linear-interpolated quantile of a non-empty ascending list."""
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _spread(vals):
    """Interquartile range of a non-empty list, or its standard deviation if that is zero."""
    s = sorted(vals)
    iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
    if iqr > 0:
        return iqr
    mu = sum(s) / len(s)
    return math.sqrt(sum((x - mu) ** 2 for x in s) / max(1, len(s) - 1))


def _numeric_values(rows, c):
    """Finite float per row for a numeric column (None where a cell is unusable),
    or None if the column is not numeric."""
    vals = [_finite(r.get(c)) for r in rows]
    present = sum(1 for x in vals if x is not None)
    if present == 0 or present < NUMERIC_FRAC * len(rows):
        return None
    return vals


def _is_id(vals):
    """True for a numeric column with almost one distinct value per row (ids, ranks)."""
    present = [x for x in vals if x is not None]
    n = len(vals)
    return n >= 20 and len(set(present)) > ID_FRAC * n


def _thin(vals, limit):
    """At most `limit` values, evenly spaced through the list."""
    if len(vals) <= limit:
        return vals
    step = len(vals) / limit
    return [vals[int(k * step)] for k in range(limit)]


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for why the good players
    in train_csv are worth signing."""
    rows, inputs, goals = _load(train_csv)
    n = len(rows)
    if n == 0:
        return "There are no training players, so no reasons can be given for signing anyone."
    if not goals:
        return ("The file has no goal columns (names ending in + or -), so no player "
                "can be called good and no reasons can be given.")
    if not inputs:
        return "The file has goals but no input columns, so there is nothing to compare."

    g = len(goals)
    oriented = [_oriented(r, goals) for r in rows]
    usable = [i for i, o in enumerate(oriented) if o is not None]
    if not usable:
        return "No player has numeric values for every goal, so players cannot be compared."
    lo, hi = _scale([oriented[i] for i in usable], g)
    score = {i: _score(oriented[i], lo, hi) for i in usable}
    usable_set = set(usable)
    good = [i for i in best(train_csv) if i in usable_set]
    if not good:
        return ("No player is unusually good on all goals together, so this data gives "
                "no evidence-based reason to sign anyone.")

    goal_names = ", ".join(c for c, _ in goals)
    lines = [
        f"Goals: {goal_names}. Each goal is scaled to 0-1 over the {len(usable)} players with "
        "complete goal values (0 = worst, 1 = best), and a player's goal score is the average.",
        f"Good players: {len(good)} of {n} rows, the Pareto front of the goals (no other player "
        "is at least as good on every goal and better on one), capped at 5% of rows.",
        "Goal medians, good players vs all players:",
    ]
    for j, (c, sign) in enumerate(goals):
        med_good = sign * _median([oriented[i][j] for i in good])
        med_all = sign * _median([oriented[i][j] for i in usable])
        lines.append(f"  {c}: {med_good:g} among good players vs {med_all:g} among all players")

    # Numeric inputs where the good players differ most from everyone.
    profile = []
    for c in inputs:
        vals = _numeric_values(rows, c)
        if vals is None or _is_id(vals):
            continue
        present = [x for x in vals if x is not None]
        spread = _spread(present)
        good_v = [vals[i] for i in good if vals[i] is not None]
        all_v = [vals[i] for i in usable if vals[i] is not None]
        if spread <= 0 or not good_v or not all_v:
            continue
        mg, ma = _median(good_v), _median(all_v)
        if mg != ma:
            profile.append((abs(mg - ma) / spread, c, mg, ma))
    profile.sort(key=lambda t: (-t[0], t[1]))
    if profile:
        lines.append("Numeric inputs where good players differ most from all players "
                     "(median among good vs all):")
        for _, c, mg, ma in profile[:TOP_PROFILE]:
            lines.append(f"  {c}: {mg:g} vs {ma:g}")

    # Most common categorical values among good players.
    cats = []
    for c in inputs:
        if _numeric_values(rows, c) is not None:
            continue
        texts = [_text(r.get(c)) for r in rows]
        if len({t for t in texts if t}) > MAX_LEVELS:
            continue
        counts = Counter(texts[i] for i in good if texts[i])
        if not counts:
            continue
        value, k = counts.most_common(1)[0]
        share = k / len(good)
        note = ("" if share >= MAJORITY
                else f"  [mixed: only {k} of {len(good)} good players agree]")
        cats.append((share, c, value, k, note))
    cats.sort(key=lambda t: (-t[0], t[1]))
    if cats:
        lines.append("Common attributes among good players (most frequent value):")
        for _, c, value, k, note in cats[:MAX_CATEGORICAL]:
            lines.append(f"  {c} = {value}  ({k} of {len(good)} good players){note}")

    # Attribute values associated with a higher goal score than the rest.
    n_use = len(usable)
    total = sum(score[i] for i in usable)
    cands = []
    screened = 0
    for c in inputs:
        texts = {i: _text(rows[i].get(c)) for i in usable}
        agg = {}
        for i in usable:
            v = texts[i]
            if not v:
                continue
            st = agg.get(v)
            if st is None:
                st = agg[v] = [0, 0.0]
            st[0] += 1
            st[1] += score[i]
        for v, (nm, sm) in agg.items():
            screened += 1
            nr = n_use - nm
            if nm < MIN_SUPPORT or nr < MIN_SUPPORT:
                continue
            mw = sm / nm
            mo = (total - sm) / nr
            if mw > mo:
                cands.append((mw - mo, c, v, mw, mo, nm, nr))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))

    if not cands:
        lines.append("No single attribute value is clearly associated with a higher goal "
                     "score in this data.")
    else:
        lines.append(
            f"Attribute values most associated with a higher goal score (of {screened} values "
            "screened; mean score with the value vs without it):"
        )
        for _, c, v, mw, mo, nm, nr in cands[:TOP_SETTINGS]:
            with_s = [score[i] for i in usable if _text(rows[i].get(c)) == v]
            without_s = [score[i] for i in usable if _text(rows[i].get(c)) != v]
            noise = same(_thin(with_s, MAX_TEST), _thin(without_s, MAX_TEST))
            verdict = ("not distinguishable from noise by a permutation test" if noise
                       else "unlikely to be noise (permutation test, p below 0.05)")
            lines.append(f"  {c} = {v}: {mw:.2f} with vs {mo:.2f} without "
                         f"(n = {nm} vs {nr}); {verdict}.")

    lines.append(
        "Caveats: these are associations in the training data, not proof that a player causes "
        "better results. Each value is judged on its own, and with many values screened some "
        "can be chance. Permutation tests use an evenly spaced sample of at most "
        f"{MAX_TEST} players per group. Check each reason on video or in fresh data before "
        "relying on it."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this player (goals included) is rated good or low."""
    rows, inputs, goals = _load(train_csv)
    if not goals:
        return "This file has no goal columns, so the player cannot be rated good or low."

    o_row = _oriented(row, goals)
    if o_row is None:
        return ("The player cannot be rated: one or more of their goal values is missing "
                "or not a number.")

    oriented = [_oriented(r, goals) for r in rows]
    usable = [i for i, o in enumerate(oriented) if o is not None]
    if not usable:
        return "The training data has no complete goal values, so the player cannot be compared."

    g = len(goals)
    lo, hi = _scale([oriented[i] for i in usable], g)
    s = _score(o_row, lo, hi)
    below = sum(1 for i in usable if _score(oriented[i], lo, hi) < s)
    pct = 100.0 * below / len(usable)
    dominators = sum(1 for i in usable if _dominates(oriented[i], o_row))
    is_good = dominators == 0

    # Goal gaps against the training median, in units of each goal's range.
    meds = [_median([oriented[i][j] for i in usable]) for j in range(g)]
    gaps = []
    for j, (_, sign) in enumerate(goals):
        span = hi[j] - lo[j]
        if span <= 0:
            continue
        gaps.append(((o_row[j] - meds[j]) / span, j, sign * o_row[j], sign * meds[j]))

    if is_good:
        verdict = (f"Rated good: no training player is at least as good on every goal and better "
                   f"on one, and the goal score {s:.2f} (0 = worst, 1 = best) is higher than "
                   f"{pct:.0f}% of training players.")
        picks = sorted((t for t in gaps if t[0] > 0), reverse=True)[:2]
        label = "strongest goals"
    else:
        verdict = (f"Rated low: {dominators} of {len(usable)} training players are at least as good "
                   f"on every goal and better on one, and the goal score {s:.2f} (0 = worst, "
                   f"1 = best) is higher than only {pct:.0f}% of training players.")
        picks = sorted(t for t in gaps if t[0] < 0)[:2]
        label = "weakest goals"

    reasons = []
    if picks:
        items = [f"{goals[j][0]} is {raw:g} against a training median of {med_raw:g}"
                 for _, j, raw, med_raw in picks]
        reasons.append(f"{label}: " + "; ".join(items))

    # Numeric inputs where this player differs most from the good players.
    usable_set = set(usable)
    good = [i for i in best(train_csv) if i in usable_set]
    if good:
        cands = []
        for c in inputs:
            vals = _numeric_values(rows, c)
            if vals is None or _is_id(vals):
                continue
            rv = _finite(row.get(c))
            if rv is None:
                continue
            good_v = [vals[i] for i in good if vals[i] is not None]
            if not good_v:
                continue
            present = [x for x in vals if x is not None]
            spread = _spread(present)
            if spread <= 0:
                continue
            mg = _median(good_v)
            diff = (rv - mg) / spread
            if abs(diff) >= MIN_DIFF:
                cands.append((abs(diff), c, rv, mg))
        cands.sort(key=lambda t: (-t[0], t[1]))
        if cands:
            parts = [f"{c} is {rv:g} against {mg:g} for good players"
                     for _, c, rv, mg in cands[:2]]
            reasons.append("numeric profile: " + "; ".join(parts))

    if reasons:
        return f"{verdict} Checkable reasons: {' and '.join(reasons)}."
    return verdict

