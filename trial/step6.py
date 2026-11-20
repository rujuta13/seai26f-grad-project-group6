"""step6.py: justify the recommended flag set and rate single builds.

explain: plain-language reasons for what makes builds good. Each claim can be
recomputed from the training csv. The recommended build is on the Pareto front.
Flag settings are compared by their share among front builds and among all
builds. A setting is reported as shifting a goal only when step3.same finds
the goal difference real.

why: one or two sentences rating a single build. A build is good when no
training build is at least as good on every goal and better on one. Its goal
values are then compared with the 10th-90th percentile range of the nearest
flag sets from step2.

Standard library only.
"""
from collections import Counter
import statistics

from step1 import _load, _columns, _dominates, best
from step2 import guess
from step3 import same
from step5 import CATEGORY_LIMIT

MIN_SHARE_GAP = 0.10   # report a setting only if its share differs by at least this
MAX_REASONS = 4        # number of flag settings checked and reported


def _goal_vectors(rows, goals):
    """Oriented goal vector per row (higher is better), or None if unusable."""
    out = []
    for r in rows:
        try:
            out.append([sgn * float(r[c]) for c, sgn in goals])
        except (KeyError, TypeError, ValueError):
            out.append(None)
    return out


def _normalised_score(vectors):
    """Return score(v): the sum over goals of v's position between the worst
    and best value seen in vectors (as in step1.best)."""
    g = len(vectors[0])
    lo = [min(v[t] for v in vectors) for t in range(g)]
    hi = [max(v[t] for v in vectors) for t in range(g)]

    def score(v):
        s = 0.0
        for t in range(g):
            span = hi[t] - lo[t]
            s += (v[t] - lo[t]) / span if span > 0 else 1.0
        return s

    return score


def _predicted(c, p):
    est, lo, hi = p
    return f"{c} median {est:g} (10th-90th percentile {lo:g} to {hi:g})"


def explain(train_csv) -> str:
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    n = len(rows)
    if n == 0 or not goals:
        return ("There is no recommended flag set: the training file has no rows "
                "or no goal columns.")

    V = _goal_vectors(rows, goals)
    usable = [i for i in range(n) if V[i] is not None]
    front = [i for i in best(train_csv) if V[i] is not None]
    if not front or not usable:
        return ("There is no recommended flag set: too few usable training builds "
                "to form a Pareto front.")

    score = _normalised_score([V[i] for i in usable])
    rec = max(front, key=lambda i: (score(V[i]), -i))
    nf = len(front)
    nu = len(usable)

    lines = [
        f"Recommended build: training row {rec} (counting from 0). It is one of "
        f"{nf} builds on the Pareto front: no training build is at least as good "
        "on every goal and better on at least one.",
        "Its settings: " + ", ".join(f"{c}={rows[rec][c]:g}" for c in inputs) + ".",
        "Goals, recommended build against all builds (medians):",
    ]
    for c, sgn in goals:
        raw_all = [float(rows[i][c]) for i in usable]
        raw_front = [float(rows[i][c]) for i in front]
        direction = "higher" if sgn > 0 else "lower"
        lines.append(
            f"- {c}: recommended {float(rows[rec][c]):g}; front median "
            f"{statistics.median(raw_front):g}; all-build median "
            f"{statistics.median(raw_all):g}; {direction} is better."
        )

    # For each input with few distinct values, find the setting whose share
    # among front builds differs most from its share among all builds.
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

    lines.append("Flag settings that set the front apart "
                 "(share among front builds vs among all builds):")
    shown = 0
    for gap, c, v, sf, sa in cands:
        if gap < MIN_SHARE_GAP or shown == MAX_REASONS:
            break
        shown += 1
        inside = [i for i in usable if rows[i][c] == v]
        outside = [i for i in usable if rows[i][c] != v]
        checks = []
        for g, _ in goals:
            xs = [float(rows[i][g]) for i in inside]
            ys = [float(rows[i][g]) for i in outside]
            real = bool(xs and ys) and not same(xs, ys)
            checks.append(f"{g} {'differs' if real else 'shows no real difference'}")
        lines.append(
            f"- {c}={v:g}: {sf:.0%} of front builds vs {sa:.0%} of all builds. "
            f"Builds with it vs without it: {'; '.join(checks)}."
        )
    if shown == 0:
        lines.append(f"- No single setting differs by {MIN_SHARE_GAP:.0%} or more "
                     "between front builds and all builds.")

    lines.append(
        "These are associations in the training data: they show where good builds "
        "sit, not that a flag causes the change. Confirm with a controlled rebuild "
        "before relying on them."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals:
        return "This training file has no goal columns, so the build cannot be rated."

    flags = {c: row[c] for c in inputs if c in row}
    pred = guess(train_csv, flags)
    got = {c: row[c] for c, _ in goals if isinstance(row.get(c), float)}

    if len(got) < len(goals):
        missing = [c for c, _ in goals if c not in got]
        return ("This build cannot be rated: it has no usable value for "
                f"{', '.join(missing)}. The nearest flag sets predict "
                + "; ".join(_predicted(c, pred[c]) for c, _ in goals) + ".")

    V = [sgn * got[c] for c, sgn in goals]
    train_vecs = [v for v in _goal_vectors(rows, goals) if v is not None]
    if not train_vecs:
        return ("This build cannot be rated: the training file has no usable "
                "builds to compare it with.")
    n = len(train_vecs)
    dominators = sum(1 for t in train_vecs if _dominates(t, V))
    if dominators == 0:
        verdict = (f"This build is good: none of the {n} training builds is at least "
                   "as good on every goal and better on at least one, so it is on "
                   "the Pareto front")
    else:
        verdict = (f"This build is bad: {dominators} of the {n} training builds are at "
                   "least as good on every goal and better on at least one, so it is "
                   "beaten")

    parts = []
    for c, sgn in goals:
        est, lo, hi = pred[c]
        o = sgn * got[c]
        olo, ohi = sorted((sgn * lo, sgn * hi))
        if o > ohi:
            rel = "is better than"
        elif o < olo:
            rel = "is worse than"
        else:
            rel = "is within"
        direction = "lower" if sgn < 0 else "higher"
        parts.append(
            f"{c} = {got[c]:g} {rel} the 10th-90th percentile range {lo:g} to {hi:g} "
            f"(median {est:g}; {direction} is better)"
        )
    return f"{verdict}. Against the nearest flag sets, " + "; ".join(parts) + "."

