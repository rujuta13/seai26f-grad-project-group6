import step1
import step2
import step5
import step16

_KEEP = 5          # most separating inputs reported by explain()
_TOP = 8           # strongest inputs tested for significance by explain()
_MIN_EFFECT = 0.2  # gaps below this are not worth a test
_ALPHA = 0.05      # family-wise level, Bonferroni-corrected over tested inputs


def _mean(xs):
    return sum(xs) / len(xs)


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes
    a group of customers worth the next campaign."""
    rows, inputs, goals = step1._load(train_csv)
    n = len(rows)
    if n == 0:
        return ("There are no customers in the training file, so there is no group "
                "to recommend and nothing to explain.")
    if not goals:
        return ("The file has no goal columns (names ending in + or -), so nothing "
                "counts as good and there is no basis for recommending a group.")

    goal_names = ", ".join(name for name, _ in goals)

    chosen = step1.best(train_csv)
    if not chosen:
        return (f"No group is recommended: the group is capped at 5% of the customers, "
                f"which rounds down to zero below 20 rows. This file has {n}.")

    scores = [step16._signed(r, goals) for r in rows]
    undominated = [
        i for i in range(n)
        if not any(step1._dominates(scores[j], scores[i])
                   for j in range(n) if j != i)
    ]
    chosen_set = set(chosen)
    rest = [i for i in range(n) if i not in chosen_set]

    lines = [
        f"The recommended group is {len(chosen)} of the {n} customers in the file. "
        f"A customer is in it only if no other customer is at least as good on every "
        f"goal ({goal_names}) and better on at least one. {len(undominated)} customers "
        f"pass that test; the group is capped at 5%, so if more pass, the ones with the "
        f"highest combined rank across goals are kept."
    ]

    for name, sign in goals:
        fv = step16._nums(rows, chosen, name)
        rv = step16._nums(rows, rest, name)
        if fv and rv:
            aim = "higher is better" if sign > 0 else "lower is better"
            lines.append(
                f"On {name} ({aim}), the group averages {_mean(fv):.4g} "
                f"(from {len(fv)} customers) and everyone else averages {_mean(rv):.4g} "
                f"(from {len(rv)})."
            )

    tested = [c for c in inputs if not step16._is_identifier(rows, c)]
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
            p = step16._perm_p(fv, rv)
            mf, mr = _mean(fv), _mean(rv)
            way = "higher" if mf > mr else "lower"
            desc = (f"{c}: the group is {way} ({mf:.3g} against {mr:.3g} "
                    f"for everyone else; standardised gap {eff:.2f})")
        else:
            v = step16._top_category(fv, rv)
            ind_f = [1.0 if x == v else 0.0 for x in fv]
            ind_r = [1.0 if x == v else 0.0 for x in rv]
            p = step16._perm_p(ind_f, ind_r)
            pf = fv.count(v) / len(fv)
            pr = rv.count(v) / len(rv)
            desc = (f"{c} = {v!r}: found in {pf:.0%} of the group against "
                    f"{pr:.0%} of everyone else")
        adj = min(1.0, p * m)
        if adj < _ALPHA:
            reported.append(
                f"{desc}; permutation p = {p:.3g}, corrected for {m} inputs = {adj:.3g}."
            )

    if reported:
        lines.append(
            "Customer attributes that set the group apart. These are the things to "
            "target on, or check, before launch:"
        )
        lines.extend("  - " + s for s in reported)
    else:
        lines.append(
            f"No attribute sets the group apart once the {m} tested inputs are corrected "
            f"for multiple testing, so the group cannot be described by any single "
            f"attribute in this file."
        )

    lines.append(
        f"These are associations within the {n} customers in the training file, not "
        f"measured effects of a campaign. The file records how customers behaved on "
        f"{goal_names}; it does not show how they would respond to a new offer. Attributes "
        f"that record earlier responses to offers may restate a goal rather than explain "
        f"it, so do not count them as separate evidence. Before the full campaign, send "
        f"the offer to this group, hold back a random control group of similar customers, "
        f"and compare responses. That test, not this summary, should decide the campaign."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this customer (goals included) is good or low value."""
    rows, inputs, goals = step1._load(train_csv)
    if not goals:
        return ("This file has no goal columns, so the customer cannot be rated good "
                "or low value.")
    desc = step16._goal_text(row, goals)
    n = len(rows)
    if n == 0:
        return (f"There are no training customers to compare with, so this customer "
                f"({desc}) cannot be rated.")

    s = step16._signed(row, goals)
    beaters = [i for i, r in enumerate(rows)
               if step1._dominates(step16._signed(r, goals), s)]
    if not beaters:
        return (f"Good: none of the {n} training customers is at least as good on every "
                f"goal and better on one ({desc}).")

    k = len(beaters)
    kinds = step2._prep(rows, inputs)
    parsed = [step2._parse(r, inputs, kinds) for r in rows]
    q = step2._parse(row, inputs, kinds)
    j = min(beaters, key=lambda i: (step2._dist(q, parsed[i], kinds), i))

    first = (f"Low value: {k} of {n} training customers are at least as good on every goal "
             f"and better on at least one ({desc}); the most similar of them, customer row "
             f"{j} ({step16._goal_text(rows[j], goals)}), is one of them.")

    gap = step16._input_gap(row, rows, beaters, inputs)
    if gap is None:
        return first
    c, _z, v, mean = gap
    way = "higher" if v > mean else "lower"
    return (first + f" Its clearest difference from those {k} is in {c}: this customer is "
            f"{way} ({v:.3g} against an average of {mean:.3g}), which is a lead to check, "
            f"not a proven cause.")

