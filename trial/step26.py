"""Step 26: justify the campaign group and the low-value rating in plain words.

explain(train_csv) describes the group that should get the next campaign: the
top customers from step21.best (or the highest mean goal score if that returns
nothing). It states how that group compares with all customers on each goal
and lists the input settings that are much more common in the group than
among all customers.

why(train_csv, row) rates one customer against the training customers and says
why. It names the goal the customer is weakest on and, when the customer is
rated low value, the input settings the customer shares with the lowest-value
training customers, so each reason can be acted on.

Every figure is a count, a share or an average computed from the training CSV,
so it can be checked by hand. These are associations in the data, not proof
that a setting causes a result.
"""
import math

from step1 import _load
from step21 import best

_TOP_FRACTION = 0.05   # share of customers treated as the top group, as in step21.best
_LOW_FRACTION = 0.05   # share of customers treated as the lowest-value reference group
_MAX_LEVELS = 8        # numeric inputs with more distinct values are skipped
_MIN_GAP = 0.20        # report a setting if it is this much more common in the reference group
_MIN_COUNT = 2         # a setting must appear in at least this many reference customers
_MAX_REASONS = 6
_MATCH_REASONS = 3
_FEW = 10              # warn when the group has fewer customers than this


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _level(v):
    """Comparable key for an input cell: finite float, non-blank string, or None."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _goal_spec(header):
    """Return [(column, sign)]: +1.0 for '+' goals, -1.0 for '-' goals."""
    out = []
    for c in header:
        if c.endswith("+"):
            out.append((c, 1.0))
        elif c.endswith("-"):
            out.append((c, -1.0))
    return out


def _oriented(row, goals):
    """Goal values with signs applied so higher is better, or None if any is missing."""
    vec = []
    for c, sg in goals:
        f = _finite(row.get(c))
        if f is None:
            return None
        vec.append(sg * f)
    return vec


def _ranges(vecs):
    g = len(vecs[0])
    lo = [min(v[j] for v in vecs) for j in range(g)]
    hi = [max(v[j] for v in vecs) for j in range(g)]
    return lo, hi


def _norm(vec, lo, hi):
    """Scale each goal to 0..1 over the training range (0 worst, 1 best), clamped."""
    out = []
    for j, x in enumerate(vec):
        span = hi[j] - lo[j]
        t = 1.0 if span == 0 else (x - lo[j]) / span
        out.append(min(1.0, max(0.0, t)))
    return out


def _scores(rows, goals):
    """Return (vecs, score, lo, hi) over the rows with every goal finite.

    vecs and score are dicts keyed by row index; score is the mean normalised
    goal value (0 worst, 1 best). lo and hi are None if no row is complete.
    """
    vecs = {}
    for i, r in enumerate(rows):
        v = _oriented(r, goals)
        if v is not None:
            vecs[i] = v
    if not vecs:
        return vecs, {}, None, None
    lo, hi = _ranges(list(vecs.values()))
    g = len(goals)
    score = {i: sum(_norm(v, lo, hi)) / g for i, v in vecs.items()}
    return vecs, score, lo, hi


def _goal_label(c, sg):
    direction = "higher is better" if sg > 0 else "lower is better"
    return f"{c[:-1]} ({direction})"


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for targeting the group.

    Names the campaign group (the top-ranked customers from step21.best, or the
    highest mean goal score if that returns nothing), compares its average on
    each goal with all customers, and lists the input settings that are at
    least _MIN_GAP more common in the group than among all customers, with the
    counts needed to recheck each one.
    """
    header, rows = _load(train_csv)
    if not rows:
        return "There are no training customers, so no campaign group can be named."
    goals = _goal_spec(header)
    if not goals:
        return ("The file has no goal columns ('+' or '-'), "
                "so customers cannot be judged valuable or not.")

    vecs, score, _, _ = _scores(rows, goals)
    if not vecs:
        return "No customer has numeric values for every goal, so no group can be named."

    ids = sorted(vecs)
    n = len(ids)
    top = [i for i in best(train_csv) if i in vecs]
    if not top:
        k = max(1, int(_TOP_FRACTION * n))
        top = sorted(ids, key=lambda i: (-score[i], i))[:k]
    grp = len(top)

    lines = [
        f"The campaign group is the {grp} top-ranked customers out of {n} with "
        "complete goal values: those that do best on all goals together."
    ]

    lines.append("Average goal results, group against all customers (raw values):")
    for c, sg in goals:
        gv = [_finite(rows[i].get(c)) for i in top]
        gv = [v for v in gv if v is not None]
        av = [_finite(rows[i].get(c)) for i in ids]
        av = [v for v in av if v is not None]
        if not gv or not av:
            continue
        lines.append(
            f"- {_goal_label(c, sg)}: group {sum(gv) / len(gv):.2f}, "
            f"all customers {sum(av) / len(av):.2f}."
        )

    findings = []
    for pos, c in enumerate(header):
        if not _is_input(c):
            continue
        cell = {i: _level(rows[i].get(c)) for i in ids}
        present = [i for i in ids if cell[i] is not None]
        grp_present = [i for i in top if cell[i] is not None]
        if not present or not grp_present:
            continue
        levels = {cell[i] for i in present}
        if all(isinstance(v, float) for v in levels) and len(levels) > _MAX_LEVELS:
            continue
        for v in levels:
            n_grp = sum(1 for i in grp_present if cell[i] == v)
            if n_grp < _MIN_COUNT:
                continue
            n_all = sum(1 for i in present if cell[i] == v)
            p_grp = n_grp / len(grp_present)
            p_all = n_all / len(present)
            if p_grp - p_all >= _MIN_GAP:
                findings.append({
                    "gap": p_grp - p_all,
                    "pos": pos,
                    "col": c,
                    "val": v,
                    "n_grp": n_grp,
                    "n_grp_total": len(grp_present),
                    "p_grp": p_grp,
                    "n_all": n_all,
                    "n_present": len(present),
                    "p_all": p_all,
                })
    findings.sort(key=lambda f: (-f["gap"], f["pos"], _fmt(f["val"])))

    if findings:
        lines.append("Settings much more common in the group than among all customers:")
        for f in findings[:_MAX_REASONS]:
            lines.append(
                f"- {f['col']} = {_fmt(f['val'])}: {f['n_grp']} of {f['n_grp_total']} "
                f"group customers ({f['p_grp']:.0%}) vs {f['n_all']} of {f['n_present']} "
                f"customers ({f['p_all']:.0%})."
            )
    else:
        lines.append(
            "No single setting is much more common in the group than among all customers."
        )

    if grp < _FEW:
        lines.append(
            f"Only {grp} customer(s) are in the group, so these shares are rough."
        )
    lines.append(
        "These are associations in the training data, not proof that a setting "
        "causes a better campaign response. Each figure can be recounted from "
        "the training file."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this customer is rated low value or not.

    The customer's mean normalised goal score is compared with the training
    customers' scores. The customer is rated low value if fewer than half of
    the training customers score no higher. The reasons name the goal the
    customer is weakest on and, for a low-value rating, the input settings the
    customer shares with the lowest-value training customers.
    """
    header, rows = _load(train_csv)
    goals = _goal_spec(header)
    if not goals:
        return "This file has no goal columns, so the customer cannot be rated."

    vecs, score, lo, hi = _scores(rows, goals)
    if not vecs:
        return ("No training customer has numeric values for every goal, "
                "so this customer cannot be rated.")

    rv = _oriented(row, goals)
    if rv is None:
        missing = [c for c, _ in goals if _finite(row.get(c)) is None]
        return ("This customer cannot be rated because goal values are missing for: "
                + ", ".join(missing) + ".")

    norm = _norm(rv, lo, hi)
    s = sum(norm) / len(goals)
    train_scores = list(score.values())
    at_most = sum(1 for t in train_scores if t <= s) / len(train_scores)
    is_low = at_most < 0.5
    verdict = "low value" if is_low else "not low value"

    j_weak = min(range(len(goals)), key=lambda j: norm[j])
    c_weak, _ = goals[j_weak]
    beats_weak = sum(1 for v in vecs.values() if v[j_weak] < rv[j_weak]) / len(vecs)
    weak_text = (
        f"{c_weak[:-1]} ({_fmt(_finite(row.get(c_weak)))}, "
        f"better than {beats_weak:.0%} of training customers)"
    )

    sentence1 = (
        f"This customer is rated {verdict}: on all goals combined they score "
        f"{s:.2f} (1 is best on every goal, 0 is worst), at least as good as "
        f"{at_most:.0%} of the {len(train_scores)} training customers."
    )

    if not is_low:
        return sentence1 + f" Its weakest goal is {weak_text}."

    ranked = sorted(score, key=lambda i: (score[i], i))
    k = max(1, int(_LOW_FRACTION * len(ranked)))
    low_ids = ranked[:k]

    reasons = []
    for pos, c in enumerate(header):
        if not _is_input(c):
            continue
        val = _level(row.get(c))
        if val is None:
            continue
        cell = {i: _level(rows[i].get(c)) for i in vecs}
        present = [i for i in vecs if cell[i] is not None]
        low_present = [i for i in low_ids if cell[i] is not None]
        if not present or not low_present:
            continue
        levels = {cell[i] for i in present}
        if all(isinstance(v, float) for v in levels) and len(levels) > _MAX_LEVELS:
            continue
        if val not in levels:
            continue
        n_low = sum(1 for i in low_present if cell[i] == val)
        if n_low < _MIN_COUNT:
            continue
        n_all = sum(1 for i in present if cell[i] == val)
        p_low = n_low / len(low_present)
        p_all = n_all / len(present)
        gap = p_low - p_all
        if gap >= _MIN_GAP:
            reasons.append((
                -gap, pos, _fmt(val), c, n_low, len(low_present), p_low, p_all,
            ))
    reasons.sort(key=lambda t: (t[0], t[1]))

    if reasons:
        parts = [
            f"{c} = {txt} ({n_low} of {n_tot} lowest-value customers, {p_low:.0%}, "
            f"vs {p_all:.0%} of all customers)"
            for _, _, txt, c, n_low, n_tot, p_low, p_all in reasons[:_MATCH_REASONS]
        ]
        match_text = (
            "it shares settings that are much more common among the lowest-value "
            "customers: " + "; ".join(parts)
        )
    else:
        match_text = (
            "it has no setting that is much more common among the lowest-value "
            "customers, so its low value comes from its goal results"
        )

    sentence2 = f"Its weakest goal is {weak_text}, and {match_text}."
    return sentence1 + " " + sentence2

