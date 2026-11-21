import math

import step1
import step3
import step5

_KEEP = 5              # most separating inputs reported by explain()
_TOP_CANDIDATES = 8    # strongest inputs tested for significance by explain()
_MIN_EFFECT = 0.2      # gaps below this are not worth a test


def _scores(row, goals):
    """Higher is better on every goal; missing values count as worst."""
    s = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        s.append(sign * v if v is not None else -math.inf)
    return tuple(s)


def _pareto(scores):
    """Indices of rows that no other row beats on every goal at once."""
    n = len(scores)
    return [
        i for i in range(n)
        if not any(step1._dominates(scores[j], scores[i])
                   for j in range(n) if j != i)
    ]


def _mean(xs):
    return sum(xs) / len(xs)


def _numbers(rows, idx, c):
    out = []
    for i in idx:
        v = step1._to_num(rows[i].get(c, ""))
        if v is not None:
            out.append(v)
    return out


def _goal_text(row, goals):
    parts = []
    for name, sign in goals:
        v = step1._to_num(row.get(name, ""))
        shown = "missing" if v is None else f"{v:.6g}"
        aim = "maximise" if sign > 0 else "minimise"
        parts.append(f"{name} {shown} ({aim})")
    return ", ".join(parts)


def _top_category(fv, rv):
    """The value whose frequency differs most between the two groups."""
    nf, nr = len(fv), len(rv)
    best_v, best_gap = None, -1.0
    for v in sorted(set(fv) | set(rv)):
        gap = abs(fv.count(v) / nf - rv.count(v) / nr)
        if gap > best_gap:
            best_v, best_gap = v, gap
    return best_v


def explain(train_csv) -> str:
    """Plain-language reasons for what makes rows good (the recommended set)."""
    rows, inputs, goals = step1._load(train_csv)
    n = len(rows)
    if n == 0:
        return "There are no training rows, so there is no recommended set to explain."
    if not goals:
        return ("The file has no goal columns (names ending in + or -), "
                "so nothing is recommended and nothing counts as good.")

    chosen = step1.best(train_csv)
    if not chosen:
        return (f"No rows are recommended: the set is capped at 5% of the rows, "
                f"which rounds down to zero below 20 rows. This file has {n}.")

    scores = [_scores(r, goals) for r in rows]
    full = _pareto(scores)
    chosen_set = set(chosen)
    rest = [i for i in range(n) if i not in chosen_set]

    lines = []
    if len(full) > len(chosen):
        lines.append(
            f"Recommended set: {len(chosen)} of {n} training rows. No other row beats "
            f"them on every goal at once, but {len(full)} rows are undominated, so the "
            f"5% cap keeps the ones with the highest combined rank across goals."
        )
    else:
        lines.append(
            f"Recommended set: the {len(chosen)} of {n} training rows that no other "
            f"training row beats on every goal at once."
        )

    for name, sign in goals:
        fv = _numbers(rows, chosen, name)
        rv = _numbers(rows, rest, name)
        aim = "higher is better" if sign > 0 else "lower is better"
        if fv and rv:
            lines.append(
                f"Goal {name} ({aim}): recommended rows average {_mean(fv):.4g}, "
                f"the rest average {_mean(rv):.4g}. A recommended row may still be "
                f"below average on one goal if it leads on another."
            )

    candidates = []
    for c in inputs:
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
            candidates.append((eff, c, kind, fv, rv))
    candidates.sort(key=lambda t: -t[0])

    reported = []
    for eff, c, kind, fv, rv in candidates[:_TOP_CANDIDATES]:
        if len(reported) >= _KEEP:
            break
        if kind == "num":
            if step3.same(fv, rv):
                continue
            mf, mr = _mean(fv), _mean(rv)
            way = "higher" if mf > mr else "lower"
            reported.append(
                f"{c}: recommended rows are {way} ({mf:.3g} against {mr:.3g} elsewhere; "
                f"standardised gap {eff:.2f}, permutation test below 5%)."
            )
        else:
            v = _top_category(fv, rv)
            ind_f = [1.0 if x == v else 0.0 for x in fv]
            ind_r = [1.0 if x == v else 0.0 for x in rv]
            if step3.same(ind_f, ind_r):
                continue
            pf = fv.count(v) / len(fv)
            pr = rv.count(v) / len(rv)
            reported.append(
                f"{c} = {v!r}: present in {pf:.0%} of recommended rows against "
                f"{pr:.0%} elsewhere (permutation test below 5%)."
            )

    if reported:
        lines.append("Inputs that separate the recommended rows from the rest:")
        lines.extend("  - " + s for s in reported)
    else:
        lines.append(
            "No single input separates the recommended rows from the rest by a clear "
            "margin, so the set is not explained by one setting."
        )

    lines.append(
        "These are associations inside the training file, not tested interventions. "
        "The strongest candidates were each checked on their own at the 5% level with "
        "no correction for how many inputs there are, so some reported settings may be "
        "chance. Confirm a setting by measuring it before relying on it."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One sentence on whether this row (goals included) is good or bad."""
    rows, _inputs, goals = step1._load(train_csv)
    if not goals:
        return "This file has no goal columns, so the row cannot be rated good or bad."
    desc = _goal_text(row, goals)
    if not rows:
        return (f"There are no training rows to compare with, so this row ({desc}) "
                f"cannot be rated.")
    s = _scores(row, goals)
    beaters = [i for i, r in enumerate(rows)
               if step1._dominates(_scores(r, goals), s)]
    if not beaters:
        return (f"Good: no training row beats it on every goal at once ({desc}), "
                f"so no training row dominates it.")
    j = beaters[0]
    return (f"Bad: {len(beaters)} training row(s) beat it on every goal at once, "
            f"for example row {j} ({_goal_text(rows[j], goals)}), so it is dominated "
            f"({desc}).")

