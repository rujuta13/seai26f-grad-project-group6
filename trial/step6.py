"""Step 6: justify the recommendation and each rating in plain words.

Standard library only. Reuses loading, column splitting and goal
normalisation from step1 (plus the dominance test), the training model and
goal estimates from step2, the recommendation from step1.best, and the
bad-row ranking from step5.

Every statement is a count or a comparison against the training data that a
reader can recompute by hand.
"""

import statistics

from step1 import _load, _columns, _normalised_goals, _dominates, best
from step2 import _model, guess
from step5 import blame, BAD_SHARE, MIN_GROUP


MIN_GAP = 0.10    # a flag setting must be this much more common among recommended builds
TOP_FLAGS = 8     # flag settings listed in explain()


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _bounds(model):
    """Per goal, (lo, hi) of the signed training values, as step1 normalises them."""
    bounds = []
    for g, (_, sign) in enumerate(model["goals"]):
        vals = [sign * t[g] for t in model["targets"]]
        bounds.append((min(vals), max(vals)))
    return bounds


def _scaled(values, goals, bounds):
    """Goal values of one build, scaled to [0, 1] with the training bounds
    (higher is better). Same rule as step1: a goal with no spread counts as 0."""
    vec = []
    for (name, sign), (lo, hi) in zip(goals, bounds):
        span = hi - lo
        vec.append((sign * values[name] - lo) / span if span > 0 else 0.0)
    return vec


def explain(train_csv) -> str:
    """Plain-words reasons for what makes the recommended builds good."""
    header, rows = _load(train_csv)
    inputs, goals = _columns(header)
    n = len(rows)
    if n == 0:
        return "There are no training builds, so nothing can be recommended."
    if not goals:
        return "The file has no goal columns (names ending in + or -), so builds cannot be ranked."

    points = _normalised_goals(rows, goals)
    n_front = sum(
        1 for i in range(n)
        if not any(_dominates(points[j], points[i]) for j in range(n) if j != i)
    )
    rec = best(train_csv)

    lines = [
        "How builds are judged: a build counts as good when no other build is at least as good "
        "on every goal and strictly better on one (the Pareto front). Goal directions: "
        + ", ".join(f"{name} ({'higher' if sign > 0 else 'lower'} is better)" for name, sign in goals)
        + ".",
        f"{n_front} of {n} training builds are on the Pareto front.",
    ]
    if not rec:
        lines.append("No build is recommended: there are too few training rows for the 5% cap.")
        return "\n".join(lines)

    lines.append(
        f"The recommendation is the {len(rec)} front builds with the highest mean normalised "
        f"goal score (at most 5% of rows)."
    )

    lines.append("Goal values, recommended builds against all builds (median, then range):")
    for name, _ in goals:
        rec_vals = [rows[i][name] for i in rec]
        all_vals = [r[name] for r in rows]
        lines.append(
            f"  {name}: recommended median {statistics.median(rec_vals):g} "
            f"(range {min(rec_vals):g} to {max(rec_vals):g}); "
            f"all builds median {statistics.median(all_vals):g} "
            f"(range {min(all_vals):g} to {max(all_vals):g})."
        )

    rec_set = set(rec)
    found = []
    for c in inputs:
        seen_all = {}
        seen_rec = {}
        for i, row in enumerate(rows):
            v = row[c]
            seen_all[v] = seen_all.get(v, 0) + 1
            if i in rec_set:
                seen_rec[v] = seen_rec.get(v, 0) + 1
        for v, k_all in seen_all.items():
            if k_all < MIN_GROUP:
                continue
            k_rec = seen_rec.get(v, 0)
            gap = k_rec / len(rec) - k_all / n
            if gap >= MIN_GAP:
                found.append((-gap, c, v, k_rec, k_all))
    found.sort(key=lambda t: (t[0], t[1], str(t[2])))

    lines.append(
        f"Flag settings over-represented among recommended builds (share among recommended "
        f"at least {MIN_GAP:.0%} above share among all builds; setting seen in at least "
        f"{MIN_GROUP} builds):"
    )
    if found:
        for _, c, v, k_rec, k_all in found[:TOP_FLAGS]:
            lines.append(
                f"  {c} = {_fmt(v)}: in {k_rec} of {len(rec)} recommended builds, "
                f"in {k_all} of {n} overall."
            )
    else:
        lines.append("  No setting stands out by this test.")

    worst = blame(train_csv)[:3]
    if worst:
        lines.append(
            f"Input columns most tied to bad builds (worst {BAD_SHARE:.0%} by goal score, step 5): "
            + ", ".join(worst) + "."
        )

    lines.append(
        "Caveats: these are associations in one training file, not tested one setting at a time. "
        "A setting that stands out may only be correlated with good results, and the ranking "
        "depends on the goals chosen and on the measurements behind them."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences: is this build good or bad, and why."""
    model = _model(train_csv)
    goals = model["goals"]
    targets = model["targets"]
    if not targets:
        return "There is no training data to compare this build with."
    if not goals:
        return "The data has no goal columns, so this build cannot be rated."

    full = dict(row)
    missing = [name for name, _ in goals if name not in row]
    if missing:
        est = guess(train_csv, row)
        for name in missing:
            full[name] = est[name][0]

    bounds = _bounds(model)
    names = [name for name, _ in goals]
    train_vecs = [_scaled(dict(zip(names, t)), goals, bounds) for t in targets]
    vec = _scaled(full, goals, bounds)

    score = sum(vec) / len(vec)
    scores = [sum(v) / len(v) for v in train_vecs]
    median = statistics.median(scores)
    good = score >= median
    beaten_by = sum(1 for p in train_vecs if _dominates(p, vec))

    weak = min(range(len(goals)), key=lambda g: vec[g])
    strong = max(range(len(goals)), key=lambda g: vec[g])
    wn = goals[weak][0]
    sn = goals[strong][0]

    note = ""
    if missing:
        note = (
            f" (goal values {', '.join(missing)} were not given and are estimated "
            f"from the nearest builds)"
        )
    s1 = (
        f"{'Good' if good else 'Bad'}: its mean goal score is {score:.2f} on a 0-to-1 scale "
        f"over the training range, {'at or above' if good else 'below'} the training median "
        f"of {median:.2f}{note}."
    )
    if beaten_by:
        dom = f"{beaten_by} training builds beat it on every goal at once, so it is dominated"
    else:
        dom = "no training build beats it on every goal at once, so it sits on the Pareto front"
    s2 = (
        f"It is weakest on {wn} ({_fmt(full[wn])}) and strongest on {sn} "
        f"({_fmt(full[sn])}), and {dom}."
    )
    return f"{s1} {s2}"

