import math
import random
from collections import Counter

import step1
import step2
import step5

_KEEP = 5          # most separating inputs reported by explain()
_TOP = 8           # strongest inputs tested for significance by explain()
_MIN_EFFECT = 0.2  # gaps below this are not worth a test
_ALPHA = 0.05      # family-wise level, Bonferroni-corrected over all tested inputs
_PERMS = 2000      # permutations per test (p-value resolution ~ 1/2001)
_SEED = 0          # fixed seed so the same inputs always give the same answer
_MIN_Z = 1.0       # in why(): smallest standardised gap worth mentioning


def _signed(row, goals):
    """Goal values signed so higher is better; missing values count as worst."""
    out = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        out.append(sign * v if v is not None else -math.inf)
    return tuple(out)


def _goal_text(row, goals):
    parts = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        shown = "missing" if v is None else f"{v:.6g}"
        aim = "maximise" if sign > 0 else "minimise"
        parts.append(f"{name} {shown} ({aim})")
    return ", ".join(parts)


def _nums(rows, idx, c):
    """Finite numeric values of column c over the rows in idx."""
    out = []
    for i in idx:
        v = step1._to_num(rows[i].get(c, ""))
        if v is not None and math.isfinite(v):
            out.append(v)
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def _is_identifier(rows, c):
    """True for columns that label rows rather than describe them.

    A column counts as an identifier when every row has a different value and
    the values are either whole numbers (row numbers, ranks) or text (names,
    links). Such columns cannot be a reason, so they are not tested.
    """
    if len(rows) <= 2:
        return False
    raw = [str(r.get(c, "")) for r in rows]
    if len(set(raw)) < len(rows):
        return False
    nums = [step1._to_num(v) for v in raw]
    if all(x is not None and x.is_integer() for x in nums):
        return True
    return all(x is None for x in nums)


def _perm_p(a, b):
    """Two-sided permutation p-value for the difference in means (add-one)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 1.0
    obs = abs(sum(a) / na - sum(b) / nb)
    if obs <= 1e-12:
        return 1.0
    pooled = list(a) + list(b)
    rng = random.Random(_SEED)
    hits = 0
    for _ in range(_PERMS):
        rng.shuffle(pooled)
        d = abs(sum(pooled[:na]) / na - sum(pooled[na:]) / nb)
        if d >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (_PERMS + 1)


def _top_category(fv, rv):
    """The value whose frequency differs most between the two groups."""
    nf, nr = len(fv), len(rv)
    cf, cr = Counter(fv), Counter(rv)
    best_v, best_gap = None, -1.0
    for v in sorted(set(cf) | set(cr), key=str):
        gap = abs(cf.get(v, 0) / nf - cr.get(v, 0) / nr)
        if gap > best_gap:
            best_v, best_gap = v, gap
    return best_v


def _input_gap(row, rows, idx, inputs):
    """The numeric input where `row` differs most from the rows in idx.

    Returns (column, z, row value, group mean) or None if no gap reaches _MIN_Z.
    """
    best = None
    for c in inputs:
        if _is_identifier(rows, c):
            continue
        v = step1._to_num(row.get(c, ""))
        if v is None or not math.isfinite(v):
            continue
        vals = _nums(rows, idx, c)
        if len(vals) < 2:
            continue
        mean = _mean(vals)
        var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
        sd = math.sqrt(var)
        if sd <= 1e-12:
            continue
        z = (v - mean) / sd
        if abs(z) >= _MIN_Z and (best is None or abs(z) > abs(best[1])):
            best = (c, z, v, mean)
    return best


def explain(train_csv) -> str:
    """Plain-language reasons for what makes rows good, for a skeptical reader."""
    rows, inputs, goals = step1._load(train_csv)
    n = len(rows)
    if n == 0:
        return "There are no training rows, so there is nothing to explain."
    if not goals:
        return ("The file has no goal columns (names ending in + or -), so nothing "
                "counts as good and there are no reasons to give.")

    goal_names = ", ".join(name for name, _ in goals)

    scores = [_signed(r, goals) for r in rows]
    undominated = [
        i for i in range(n)
        if not any(step1._dominates(scores[j], scores[i])
                   for j in range(n) if j != i)
    ]

    chosen = step1.best(train_csv)
    if not chosen:
        return (f"No rows are recommended: the set is capped at 5% of the rows, which "
                f"rounds down to zero below 20 rows. This file has {n}.")
    chosen_set = set(chosen)
    rest = [i for i in range(n) if i not in chosen_set]

    lines = [
        f"A row counts as good here if no other row is at least as good on every goal "
        f"({goal_names}) and better on at least one. {len(undominated)} of the {n} rows "
        f"meet that test. The {len(chosen)} recommended rows are capped at 5% of the "
        f"file; if more rows pass the test, the ones with the highest combined rank "
        f"across goals are kept."
    ]

    for name, sign in goals:
        fv = _nums(rows, chosen, name)
        rv = _nums(rows, rest, name)
        if fv and rv:
            aim = "higher is better" if sign > 0 else "lower is better"
            lines.append(
                f"{name} ({aim}): recommended rows average {_mean(fv):.4g} "
                f"(from {len(fv)} values), the rest average {_mean(rv):.4g} "
                f"(from {len(rv)}). A recommended row can sit below average on one "
                f"goal because it was chosen on the goals together."
            )

    tested = [c for c in inputs if not _is_identifier(rows, c)]
    m = max(1, len(tested))

    candidates = []
    for c in tested:
        kind, vals = step5._column_values(rows, c)
        fv = [vals[i] for i in chosen]
        rv = [vals[i] for i in rest]
        if not fv or not rv:
            continue
        if kind == "num":
            if len(fv) < 2 or len(rv) < 2:
                continue
            eff = step5._numeric_effect(fv, rv)
        else:
            eff = step5._categorical_effect(fv, rv)
        if eff >= _MIN_EFFECT:
            candidates.append((eff, c, kind, vals))
    candidates.sort(key=lambda t: (-t[0], t[1]))

    reported = []
    for eff, c, kind, vals in candidates[:_TOP]:
        if len(reported) >= _KEEP:
            break
        fv = [vals[i] for i in chosen]
        rv = [vals[i] for i in rest]
        if kind == "num":
            p = _perm_p(fv, rv)
            mf, mr = _mean(fv), _mean(rv)
            way = "higher" if mf > mr else "lower"
            desc = (f"{c}: recommended rows are {way} ({mf:.3g} against {mr:.3g} "
                    f"elsewhere; standardised gap {eff:.2f})")
        else:
            v = _top_category(fv, rv)
            ind_f = [1.0 if x == v else 0.0 for x in fv]
            ind_r = [1.0 if x == v else 0.0 for x in rv]
            p = _perm_p(ind_f, ind_r)
            pf = fv.count(v) / len(fv)
            pr = rv.count(v) / len(rv)
            desc = (f"{c} = {v!r}: present in {pf:.0%} of recommended rows against "
                    f"{pr:.0%} elsewhere")
        adj = min(1.0, p * m)
        if adj < _ALPHA:
            reported.append(
                f"{desc}; permutation p = {p:.3g}, corrected for {m} inputs = {adj:.3g}."
            )

    if reported:
        lines.append("Attributes that separate the recommended rows from the rest. "
                     "Where they are visible on video, they are concrete things to check:")
        lines.extend("  - " + s for s in reported)
    else:
        lines.append(
            f"No attribute separates the recommended rows from the rest once the {m} "
            f"tested inputs are corrected for multiple testing, so the set is not "
            f"explained by any single attribute in this file."
        )

    lines.append(
        f"These are associations within the {n} rows of the training file, not tested "
        f"interventions. The file measures only {goal_names}; nothing outside those goals "
        f"is judged. A difference here is a reason to look, not proof of how any player "
        f"will perform, and this summary does not by itself establish that a particular "
        f"player should be signed."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on whether this row (goals included) is good or bad."""
    rows, inputs, goals = step1._load(train_csv)
    if not goals:
        return "This file has no goal columns, so the row cannot be rated good or bad."
    desc = _goal_text(row, goals)
    n = len(rows)
    if n == 0:
        return (f"There are no training rows to compare with, so this row ({desc}) "
                f"cannot be rated.")

    s = _signed(row, goals)
    beaters = [i for i, r in enumerate(rows)
               if step1._dominates(_signed(r, goals), s)]
    if not beaters:
        return (f"Good: none of the {n} training rows is at least as good on every goal "
                f"and better on one ({desc}).")

    k = len(beaters)
    kinds = step2._prep(rows, inputs)
    parsed = [step2._parse(r, inputs, kinds) for r in rows]
    q = step2._parse(row, inputs, kinds)
    j = min(beaters, key=lambda i: (step2._dist(q, parsed[i], kinds), i))

    first = (f"Bad: {k} of {n} training rows beat it on every goal ({desc}); the most "
             f"similar of them, row {j} ({_goal_text(rows[j], goals)}), is at least as "
             f"good on each goal and better on at least one.")

    gap = _input_gap(row, rows, beaters, inputs)
    if gap is None:
        return first
    c, _z, v, mean = gap
    way = "higher" if v > mean else "lower"
    return (first + f" Among those {k} rows, its clearest difference is in {c}: this row "
            f"is {way} ({v:.3g} against an average of {mean:.3g}), which is worth "
            f"checking on video.")

