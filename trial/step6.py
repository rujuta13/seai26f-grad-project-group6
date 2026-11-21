"""Step 6: justify the recommended flag set and single builds in plain words.

explain() reports which goal values the non-dominated builds reach and which
input settings they share, with the counts behind each claim so a skeptic can
recompute them. why() rates one build against the training data and states the
two facts that drive the rating.
"""

import math

from step1 import _load, _split_columns, best
from step2 import _quantile, guess
from step3 import same
from step5 import _as_float, _column_labels, blame

MIN_SUPPORT = 5
TOP_BLAME = 3


def _median(vals):
    return _quantile(sorted(vals), 0.5)


def _dominates(a, b):
    """True if a Pareto-dominates b (all dimensions minimised)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _min_vector(values, maximise, minimise):
    """Goal values as a minimisation vector ('+' negated), or None if any is missing."""
    vec = []
    for c in maximise:
        v = _as_float(values.get(c))
        if v is None:
            return None
        vec.append(-v)
    for c in minimise:
        v = _as_float(values.get(c))
        if v is None:
            return None
        vec.append(v)
    return vec


def _normaliser(vecs):
    """Return a function scaling a minimisation vector by the range of vecs."""
    m = len(vecs[0])
    lo = [min(v[j] for v in vecs) for j in range(m)]
    hi = [max(v[j] for v in vecs) for j in range(m)]

    def norm(v):
        return [
            (v[j] - lo[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else 0.0
            for j in range(m)
        ]

    return norm


def _label_text(label):
    if isinstance(label, int) and not isinstance(label, bool):
        return f"quantile bin {label} of this column (0 = lowest values)"
    return f"value {label}"


def explain(train_csv) -> str:
    """Plain-language reasons that make rows good, with the counts behind them."""
    columns, rows = _load(train_csv)
    maximise, minimise, _ = _split_columns(columns)
    goals = maximise + minimise
    n = len(rows)
    if n == 0 or not goals:
        return "No goal columns or no rows, so there is nothing to justify."

    good = set(best(train_csv))
    if not good:
        return "No build is non-dominated on the goals, so no recommended set exists to explain."
    rest = [i for i in range(n) if i not in good]

    lines = [
        f"Of {n} measured builds, {len(good)} are on the goal Pareto front: no other build "
        "is at least as good on every goal and strictly better on at least one. "
        "These are the recommended builds; the remaining builds are the comparison group."
    ]

    if rest:
        goal_lines = []
        good_sorted = sorted(good)
        for g in goals:
            gv = [v for v in (_as_float(rows[i].get(g)) for i in good_sorted) if v is not None]
            rv = [v for v in (_as_float(rows[i].get(g)) for i in rest) if v is not None]
            if not gv or not rv:
                continue
            verdict = (
                "significant difference" if not same(gv, rv) else "no significant difference"
            )
            goal_lines.append(
                f"  {g}: median {_median(gv):.6g} vs {_median(rv):.6g} ({verdict}, "
                "permutation test at the 5% level)."
            )
        if goal_lines:
            lines.append("Goal values, recommended builds vs the rest:")
            lines.extend(goal_lines)
            lines.append(
                "This comparison favours the recommended builds by construction, so treat it "
                "as a check on the selection, not as evidence that the settings cause the goals."
            )

        ng, nr = len(good), len(rest)
        setting_lines = []
        for c in blame(train_csv)[:TOP_BLAME]:
            labels = _column_labels(rows, c)
            g_counts, r_counts = {}, {}
            for i in range(n):
                tgt = g_counts if i in good else r_counts
                tgt[labels[i]] = tgt.get(labels[i], 0) + 1

            best_lab, best_gain, best_pair = None, 0.0, None
            for lab, cnt in g_counts.items():
                if cnt < MIN_SUPPORT:
                    continue
                gs = cnt / ng
                rs = r_counts.get(lab, 0) / nr
                if gs - rs > best_gain:
                    best_lab, best_gain, best_pair = lab, gs - rs, (gs, rs)
            if best_lab is None:
                continue
            gs, rs = best_pair
            setting_lines.append(
                f"  {c} = {_label_text(best_lab)}: {gs:.0%} of recommended builds "
                f"vs {rs:.0%} of the rest."
            )

        if setting_lines:
            lines.append("Input settings that most separate the recommended builds from the rest:")
            lines.extend(setting_lines)
        else:
            lines.append(
                "No single input setting is shared by at least "
                f"{MIN_SUPPORT} recommended builds much more often than by the rest."
            )

    lines.append(
        "These are associations in this data set; confirm any single setting by building "
        "with and without it before relying on it."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences saying whether this build is good or bad, and why."""
    columns, rows = _load(train_csv)
    maximise, minimise, _ = _split_columns(columns)
    goals = maximise + minimise
    if not rows or not goals:
        return "Cannot rate this build: the training data has no rows or no goal columns."

    train_vecs = []
    for r in rows:
        v = _min_vector(r, maximise, minimise)
        if v is not None and all(math.isfinite(x) for x in v):
            train_vecs.append(v)
    if not train_vecs:
        return "Cannot rate this build: no training row has complete, finite goal values."

    measured = {g: _as_float(row.get(g)) for g in goals}
    if all(v is not None for v in measured.values()):
        values = measured
        source = "measured"
    else:
        est = guess(train_csv, row)
        values = {g: est[g][0] for g in goals}
        source = "estimated from the most similar training builds"

    norm = _normaliser(train_vecs)
    tn = [norm(v) for v in train_vecs]
    q = norm(_min_vector(values, maximise, minimise))

    q_score = sum(q)
    t_scores = [sum(t) for t in tn]
    threshold = _quantile(sorted(t_scores), 0.75)
    is_good = q_score <= threshold
    pct_worse = 100.0 * sum(1 for s in t_scores if s > q_score) / len(t_scores)
    dominators = sum(1 for t in tn if _dominates(t, q))

    goal_txt = ", ".join(f"{g}={values[g]:.6g}" for g in goals)
    verdict = "good" if is_good else "bad"
    s1 = f"Rated {verdict}: its {source} goals are {goal_txt}."

    if dominators == 0:
        dom = "no training build is at least as good on every goal and strictly better on one"
    else:
        dom = (
            f"{dominators} of {len(tn)} training builds are at least as good on every goal "
            "and strictly better on one"
        )
    s2 = (
        f"Its normalised goal sum (lower is better) is better than {pct_worse:.0f}% of the "
        f"{len(tn)} training builds, and {dom}."
    )
    return f"{s1} {s2}"

