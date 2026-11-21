"""step26.py: justify the campaign group and rate single customers.

explain(train_csv): reasons a sales director can check in the customer data for
why a group should get the next campaign. The group is the best-customer front
from step21.best: customers that no other customer is at least as good as on
every goal and better than on one, capped and ranked as in step1.best. The group
is described by its goal medians against all customers, by the customer
segments (step21.families) that hold it, and by the flag settings that set it
apart. A setting is reported as shifting a goal only when step23.same finds that
goal difference real.

why(train_csv, row): one or two sentences on whether a customer is low value. A
customer is low value when some training customer is at least as good on every
goal and better on one. The reasons name the goals on which the customer falls
below the calibrated 10th-90th percentile range from step12.guess, so the sales
team knows what to work on.

Reuses step1 loading and column roles, step5 category limit, step6 share rules,
step11 fast front, step12 calibrated guesser, step14 orientation, step16 format
and sampling helpers, step21 segments and step23 certification. Standard library only.
"""
import math
import statistics
from collections import Counter

from step1 import _load, _columns, _dominates, BEST_FRACTION
from step5 import CATEGORY_LIMIT
from step6 import MIN_SHARE_GAP, MAX_REASONS, _predicted
from step11 import _pareto_front, _top_by_score
from step12 import guess
from step14 import _orient
from step16 import _fmt, _sample, SAMPLE_SEED
from step21 import families
from step23 import same


def explain(train_csv) -> str:
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    n = len(rows)
    if n == 0 or not goals:
        return ("There is no campaign group to recommend: the customer file has no "
                "rows or no goal columns.")

    V = [_orient(r, goals) for r in rows]
    usable = [i for i in range(n) if V[i] is not None]
    if not usable:
        return ("There is no campaign group to recommend: no customer has usable "
                "values for every goal.")

    # Best-customer front, as in step1.best and step11.best, on the usable rows.
    cap = max(1, int(BEST_FRACTION * n))
    UV = [V[i] for i in usable]
    front_pos = _pareto_front(UV)
    if len(front_pos) > cap:
        front_pos = _top_by_score(UV, front_pos, cap)
    front = sorted(usable[p] for p in front_pos)
    if not front:
        return ("There is no campaign group to recommend: too few usable customers "
                "to form a best-customer front.")

    nf = len(front)
    nu = len(usable)

    lines = [
        f"The campaign group is the best-customer front: {nf} of {nu} usable "
        "customers. No other customer is at least as good on every goal and better "
        "on at least one of them, so these are the customers who do best overall.",
        "Goals, the campaign group against all customers (medians):",
    ]
    for c, sgn in goals:
        raw_all = [float(rows[i][c]) for i in usable]
        raw_front = [float(rows[i][c]) for i in front]
        direction = "higher" if sgn > 0 else "lower"
        lines.append(
            f"- {c}: group median {statistics.median(raw_front):g}; all-customer "
            f"median {statistics.median(raw_all):g}; {direction} is better."
        )

    # Segments: which customer clusters hold more of the group than their size.
    labels = families(train_csv)
    size = Counter(labels[i] for i in usable)
    in_group = Counter(labels[i] for i in front)
    share = {k: in_group.get(k, 0) / size[k] for k in size}
    overall = nf / nu
    ranked = sorted(size, key=lambda k: (-share[k], k))
    strong = [k for k in ranked if share[k] > overall][:MAX_REASONS]
    lines.append(
        "Customer segments (clusters of similar flags), against the group's overall "
        f"share of {overall:.0%}:"
    )
    if not strong:
        lines.append("- No segment holds a larger share of the group than of all "
                     "customers.")
    for k in strong:
        lines.append(
            f"- Segment {k}: {in_group.get(k, 0)} of its {size[k]} customers "
            f"({share[k]:.0%}) are in the group."
        )

    # For each input with few distinct values, find the setting whose share
    # among group customers differs most from its share among all customers.
    cands = []
    for c in inputs:
        count_all = Counter(rows[i][c] for i in usable)
        if not count_all or len(count_all) > CATEGORY_LIMIT:
            continue  # continuous input: no shares to compare
        count_front = Counter(rows[i][c] for i in front)
        gap, v, sf, sa = max(
            ((abs(count_front.get(v, 0) / nf - cnt / nu),
              v, count_front.get(v, 0) / nf, cnt / nu)
             for v, cnt in count_all.items()),
            key=lambda t: t[0],
        )
        cands.append((gap, c, v, sf, sa))
    cands.sort(key=lambda t: -t[0])

    lines.append("Flag settings that set the group apart "
                 "(share among group customers vs among all customers):")
    shown = 0
    for gap, c, v, sf, sa in cands:
        if gap < MIN_SHARE_GAP or shown == MAX_REASONS:
            break
        shown += 1
        inside = [i for i in usable if rows[i][c] == v]
        outside = [i for i in usable if rows[i][c] != v]
        checks = []
        for g, _ in goals:
            xs = _sample([float(rows[i][g]) for i in inside], SAMPLE_SEED)
            ys = _sample([float(rows[i][g]) for i in outside], SAMPLE_SEED)
            real = bool(xs and ys) and not same(xs, ys)
            checks.append(f"{g} {'differs' if real else 'shows no real difference'}")
        lines.append(
            f"- {c}={_fmt(v)}: {sf:.0%} of group customers vs {sa:.0%} of all "
            f"customers. Customers with it vs without it: {'; '.join(checks)}."
        )
    if shown == 0:
        lines.append(f"- No single setting differs by {MIN_SHARE_GAP:.0%} or more "
                     "between group customers and all customers.")

    lines.append(
        "These are associations in the training data: they show what the best "
        "customers look like, not that a campaign will cause a response. Test the "
        "group with a small held-out campaign before scaling it up."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals:
        return "This file has no goal columns, so the customer cannot be rated."

    pred = guess(train_csv, row)
    V = _orient(row, goals)
    if V is None:
        missing = [c for c, _ in goals
                   if not isinstance(row.get(c), float) or not math.isfinite(row[c])]
        predicted = "; ".join(_predicted(c, pred[c]) for c, _ in goals)
        return ("This customer cannot be rated: no usable value for "
                f"{', '.join(missing)}. Their most similar training customers predict "
                f"{predicted}.")

    train_vecs = [v for v in (_orient(r, goals) for r in rows) if v is not None]
    if not train_vecs:
        return ("This customer cannot be rated: the training file has no usable "
                "customers to compare with.")
    n = len(train_vecs)
    dominators = sum(1 for t in train_vecs if _dominates(t, V))
    if dominators == 0:
        verdict = (f"This customer is not low value: none of the {n} training "
                   "customers is at least as good on every goal and better on one")
    else:
        verdict = (f"This customer is low value: {dominators} of the {n} training "
                   "customers are at least as good on every goal and better on one")

    parts = []
    shortfall = []
    for c, sgn in goals:
        est, lo, hi = pred[c]
        value = float(row[c])
        o = sgn * value
        olo, ohi = sorted((sgn * lo, sgn * hi))
        if o > ohi:
            rel = "is better than"
        elif o < olo:
            rel = "is worse than"
            shortfall.append(c)
        else:
            rel = "is within"
        direction = "lower" if sgn < 0 else "higher"
        parts.append(
            f"{c} = {value:g} {rel} the calibrated range {lo:g} to {hi:g} "
            f"(median {est:g}; {direction} is better)"
        )

    work_on = (", ".join(shortfall) if shortfall
               else "none, as no goal sits below its calibrated range")
    return (f"{verdict}. Against the nearest training customers, "
            + "; ".join(parts) + f"; goals to work on: {work_on}.")

