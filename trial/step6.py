"""Step 6: justify. Plain-language reasons a skeptical engineer can check."""

import math

import step1
import step3
import step5

_MIN_GAP = 0.2     # smallest gap (share difference, or mean difference as a fraction of
                   # the input's range) that counts as a reason
_MAX_REASONS = 6   # cap on reasons listed by explain()


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


def _is_binary(rows, name):
    """True if the input column holds only 0.0 and 1.0 (flags)."""
    vals = [r.get(name) for r in rows]
    finite = [v for v in vals if _is_num(v)]
    return (bool(finite) and all(isinstance(v, float) for v in vals)
            and set(finite) <= {0.0, 1.0})


def _fmt(v):
    """Short, readable rendering of a cell value."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def explain(train_csv) -> str:
    """Plain-words reasons for what makes rows good.

    Good rows are the step1.best set (at most 5% of training rows). Each input is
    compared between good rows and the rest; large gaps become reasons. Goal
    differences are checked with a permutation test (step3.same). Inputs most
    tied to bad goal values come from step5.blame.
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    goals = goals_max + goals_min
    if not rows or not goals:
        return ("No recommendation: the data needs at least one row and at least one "
                "goal column (names ending in + or -).")
    order = step1.best(train_csv)
    if not order:
        return "No recommendation: no row could be ranked on the goals."

    good_set = set(order)
    good = [rows[i] for i in order]
    rest = [rows[i] for i in range(len(rows)) if i not in good_set]

    top = rows[order[0]]
    on = [n for n in inputs if _is_binary(rows, n) and top.get(n) == 1.0]
    other = [f"{n}={_fmt(top.get(n))}" for n in inputs if not _is_binary(rows, n)]
    measured = [f"{g}={_fmt(top.get(g))}" for g in goals]
    lines = [
        f"Recommended row is training row {order[0]}: it turns on "
        f"{', '.join(on) if on else 'no 0/1 flags'}"
        + (f" and sets {', '.join(other)}" if other else "")
        + f". Its measured goals are {', '.join(measured)}."
    ]

    reasons = []  # (negative strength, input position, sentence)
    for idx, name in enumerate(inputs):
        vals = [r.get(name) for r in rows]
        if all(isinstance(v, float) for v in vals):
            gv = [r[name] for r in good if _is_num(r.get(name))]
            rv = [r[name] for r in rest if _is_num(r.get(name))]
            allv = [v for v in vals if _is_num(v)]
            if not gv or not rv or not allv:
                continue
            span = max(allv) - min(allv)
            if span <= 0:
                continue
            mg = sum(gv) / len(gv)
            mr = sum(rv) / len(rv)
            gap = (mg - mr) / span
            if abs(gap) < _MIN_GAP:
                continue
            if _is_binary(rows, name):
                if gap > 0:
                    text = (f"flag {name} is on in {mg:.0%} of good rows "
                            f"versus {mr:.0%} of the rest")
                else:
                    text = (f"flag {name} is off in {1 - mg:.0%} of good rows "
                            f"versus {1 - mr:.0%} of the rest")
            else:
                text = (f"{name} averages {mg:.4g} in good rows "
                        f"versus {mr:.4g} in the rest")
            reasons.append((-abs(gap), idx, text))
        else:
            goodstr = [str(r.get(name)) for r in good]
            reststr = [str(r.get(name)) for r in rest]
            best_gap, best_text = 0.0, None
            for c in sorted(set(goodstr) | set(reststr)):
                sg = goodstr.count(c) / len(goodstr)
                sr = reststr.count(c) / len(reststr)
                if abs(sg - sr) > abs(best_gap):
                    best_gap = sg - sr
                    best_text = (f"{name}={c} in {sg:.0%} of good rows "
                                 f"versus {sr:.0%} of the rest")
            if best_text is not None and abs(best_gap) >= _MIN_GAP:
                reasons.append((-abs(best_gap), idx, best_text))

    reasons.sort()
    if reasons:
        lines.append("Reasons the good rows differ from the rest, strongest first:")
        for _, _, text in reasons[:_MAX_REASONS]:
            lines.append(f"- {text}.")
    else:
        lines.append(f"No single input separates the good rows from the rest "
                     f"by at least {_MIN_GAP:.0%}.")

    for g in goals:
        gv = [r[g] for r in good if _is_num(r.get(g))]
        rv = [r[g] for r in rest if _is_num(r.get(g))]
        if not gv or not rv:
            continue
        mg = sum(gv) / len(gv)
        mr = sum(rv) / len(rv)
        real = not step3.same(gv, rv)
        verdict = ("this gap is real under a permutation test."
                   if real else "this gap could be noise.")
        lines.append(f"{g}: good rows average {mg:.4g} versus {mr:.4g} elsewhere; "
                     + verdict)

    blamed = step5.blame(train_csv)[:3]
    if blamed:
        lines.append("Inputs most tied to bad goal values (step5): "
                     + ", ".join(blamed) + ".")
    lines.append("These are associations in this one benchmark set, not proof of "
                 "causes; to check a flag, rebuild with only that flag changed and "
                 "compare the goals.")
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this row (goals included) is good or bad.

    Per goal, the share of training rows the row is at least as good as is
    computed, direction-aware (minimised goals are negated). Missing goal
    values count as worst. The row is good if its average share exceeds 0.5.
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not rows or not goals:
        return ("Cannot rate this row: the training data needs at least one row "
                "and at least one goal column.")

    scored = []  # (goal, share of training builds this row is at least as good as)
    for g, sign in goals:
        pool = [t[g] * sign for t in rows if _is_num(t.get(g))]
        if not pool:
            continue
        v = row.get(g)
        if _is_num(v):
            x = v * sign
            share = sum(1 for p in pool if p <= x) / len(pool)
        else:
            share = 0.0  # missing counts as worst
        scored.append((g, share))
    if not scored:
        return "Cannot rate this row: no training values exist for any goal."

    overall = sum(s for _, s in scored) / len(scored)
    verdict = "good" if overall > 0.5 else "bad"
    detail = "; ".join(f"{g} {s:.0%}" for g, s in scored)
    first = (f"Rated {verdict}: on average it is at least as good as {overall:.0%} "
             f"of training builds (per goal: {detail}).")

    second = ""
    best_rows = step1.best(train_csv)
    if best_rows and inputs:
        ref = rows[best_rows[0]]
        agree = sum(1 for n in inputs
                    if row.get(n) is not None and row.get(n) == ref.get(n))
        second = (f" Its input settings match the recommended row "
                  f"(training row {best_rows[0]}) on {agree} of {len(inputs)} inputs.")
    return first + second

