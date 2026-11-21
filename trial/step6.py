"""step6.py: plain-language justification of the recommended flag set and of single-build ratings."""

import numpy as np

from step1 import _load, best
from step2 import guess
from step3 import same
from step4 import _goal_matrix
from step5 import blame, _goal_scores


def _num(v):
    """Return v as a finite float, or None if it is not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _flag(v):
    """Readable form of a setting: 0/1 become off/on, other values are shown as they are."""
    f = _num(v)
    if f == 1.0:
        return "on"
    if f == 0.0:
        return "off"
    return str(v)


def explain(train_csv) -> str:
    """Plain-words reasons for what makes builds good, checkable by a skeptic."""
    rows, inputs, goals = _load(train_csv)
    if not rows or not goals:
        return ("There are no measured builds with goal columns, "
                "so no recommendation can be made.")

    n = len(rows)
    names = [name for name, _ in goals]

    # Combined score: each goal z-scored so that all goals count equally.
    score = _goal_scores(rows, goals)
    top = int(np.argmax(score))

    # Pareto front (possibly the ranked top 5% subset when the front is large).
    good_idx = sorted(set(best(train_csv)))
    good_set = set(good_idx)
    rest_idx = [i for i in range(n) if i not in good_set]

    lines = []
    lines.append(
        f"Recommended build: row {top} of {n} measured builds. It has the highest "
        "combined score, where each goal is z-scored (standardised) so that goals "
        "measured in different units count equally, and minimised goals are negated."
    )
    if top in good_set:
        lines.append(
            "It is also among the good builds: no measured build is at least as good "
            "on every goal and better on one."
        )
    else:
        lines.append(
            "It is not among the good builds, so some measured build beats it on every "
            "goal at once. The combined score is a trade-off, not a dominance claim."
        )
    lines.append(
        "Raw goal values: "
        + ", ".join(f"{name}={rows[top][name]}" for name in names)
        + "."
    )

    if rest_idx:
        lines.append(
            f"Settings of the recommended build compared with the {len(good_idx)} good "
            f"builds and the {len(rest_idx)} others (permutation test at 5%):"
        )
    else:
        lines.append(
            f"Settings of the recommended build, as used by the {len(good_idx)} good builds:"
        )

    # Show the input columns most associated with bad rows first, then compare settings.
    blamed = blame(train_csv)
    for c in blamed[:6]:
        v = str(rows[top][c])
        g = [1.0 if str(rows[i][c]) == v else 0.0 for i in good_idx]
        o = [1.0 if str(rows[i][c]) == v else 0.0 for i in rest_idx]
        share_g = float(np.mean(g)) if g else 0.0
        share_o = float(np.mean(o)) if o else 0.0
        verdict = "difference looks real" if not same(g, o) else "difference looks like noise"
        lines.append(
            f"- {c} = {_flag(v)}: used by {share_g:.0%} of good builds versus "
            f"{share_o:.0%} of the others ({verdict})."
        )

    lines.append(
        "Caveats: these are associations in measured data, not proof that a flag "
        "causes the result. Flags interact, so a setting can help in one combination "
        "and hurt in another. The recommended build is a measured row, not a prediction."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this build (goals included) is good or bad."""
    rows, inputs, goals = _load(train_csv)
    if not goals:
        return "No goal columns are defined, so this build cannot be rated good or bad."
    if not rows:
        return "There are no training builds to compare against, so no rating is possible."

    names = [name for name, _ in goals]
    ref = _goal_matrix(rows, goals)  # larger is always better

    measured = all(name in row and _num(row[name]) is not None for name in names)
    if measured:
        v = _goal_matrix([row], goals)[0]
        ge = (ref >= v).all(axis=1)
        gt = (ref > v).any(axis=1)
        dom = np.flatnonzero(ge & gt)
        if dom.size == 0:
            return ("Good: no training build is at least as good on every goal and better "
                    "on one, so nothing beats it outright and it sits on the Pareto front.")
        # Name the dominating build that is best overall, and the goals where it is ahead.
        j = int(dom[np.argmax(ref[dom].sum(axis=1))])
        lagging = [
            f"{names[k]} ({rows[j][names[k]]} vs {row[names[k]]})"
            for k in range(len(names))
            if ref[j, k] > v[k]
        ]
        return (f"Bad: {dom.size} training build(s) are at least as good on every goal, and "
                f"build {j} beats this one on {', '.join(lagging)}.")

    # Goals missing: predict them and judge the estimates against the training builds.
    est = guess(train_csv, row)
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    e = np.array(
        [float(est[name][0]) * sign for name, sign in goals],
        dtype=float,
    )
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(((e - mu) / sd).sum())
    if total > 0:
        return ("Predicted good (goals not measured): the model's estimates are, on balance, "
                "better than the average training build across the goals.")
    return ("Predicted bad (goals not measured): the model's estimates are, on balance, "
            "worse than the average training build across the goals.")

