"""Step 16: justify. Two client questions, answered in plain words.

Standard library only. Reuses loading, column splitting and goal normalisation
from step1, the training model and goal estimates from step2, the explanation,
scoring and bounds helpers from step6, and the per-goal blame helpers from
step15. Only the per-goal causes and the weakest-goal sentence are new.
"""

import statistics

from step1 import _load, _columns, _normalised_goals, _dominates
from step2 import _model, guess
from step5 import BAD_SHARE, MIN_GROUP
from step6 import explain as _explain, _bounds, _scaled, _fmt
from step15 import _bad_rows, _column_blame


def _worst_value(rows, col, bad):
    """(value, bad count, count) for the value of col with the highest share of
    bad rows, among values seen in at least MIN_GROUP rows. None if there is none."""
    groups = {}
    for i, row in enumerate(rows):
        g = groups.setdefault(row[col], [0, 0])
        g[0] += 1
        if i in bad:
            g[1] += 1
    ranked = sorted(
        ((v, cnt, bc) for v, (cnt, bc) in groups.items() if cnt >= MIN_GROUP),
        key=lambda t: (-t[2] / t[1], str(t[0])),
    )
    return ranked[0] if ranked else None


def _goal_causes(rows, inputs, goals):
    """For each goal, the input column and value most tied to the worst builds
    on that goal, as step 15 measures it, with counts a reader can recount."""
    n = len(rows)
    n_bad = max(1, int(n * BAD_SHARE))
    base = n_bad / n
    norm = _normalised_goals(rows, goals)
    lines = [
        f"Per goal, the worst {BAD_SHARE:.0%} of builds on that goal are checked "
        f"against each input column (step 15):"
    ]
    for g, (name, _) in enumerate(goals):
        bad = _bad_rows([p[g] for p in norm], n_bad)
        blames = [
            (_column_blame(rows, c, bad, n, base), k, c)
            for k, c in enumerate(inputs)
        ]
        top_blame, _, top_col = max(blames, key=lambda t: (t[0], -t[1]))
        if top_blame <= 0:
            lines.append(f"  {name}: no input setting stands out.")
            continue
        worst = _worst_value(rows, top_col, bad)
        if worst is None:
            lines.append(
                f"  {name}: most tied to {top_col}, but no value of it is seen "
                f"often enough to name."
            )
            continue
        v, cnt, bc = worst
        lines.append(
            f"  {name}: most tied to {top_col}; {bc} of the {cnt} builds with "
            f"{top_col} = {_fmt(v)} are in the worst {BAD_SHARE:.0%} on {name}, "
            f"against {n_bad} of {n} builds overall."
        )
    return lines


def explain(train_csv) -> str:
    """Plain-words reasons for the manager: what makes builds good, and on each
    goal which setting is most tied to the weakest builds."""
    text = _explain(train_csv)
    header, rows = _load(train_csv)
    inputs, goals = _columns(header)
    if not rows or not inputs or not goals:
        return text
    return text + "\n" + "\n".join(_goal_causes(rows, inputs, goals))


def why(train_csv, row) -> str:
    """Two sentences: whether this build (goals included) is good or bad, and
    its weakest goal, checkable against the training builds."""
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

    names = [name for name, _ in goals]
    bounds = _bounds(model)
    train_vecs = [_scaled(dict(zip(names, t)), goals, bounds) for t in targets]
    vec = _scaled(full, goals, bounds)

    score = sum(vec) / len(vec)
    median = statistics.median([sum(v) / len(v) for v in train_vecs])
    good = score >= median

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

    weak = min(range(len(goals)), key=lambda g: vec[g])
    wn, wsign = goals[weak]
    direction = "higher" if wsign > 0 else "lower"
    better_on_weak = sum(1 for v in train_vecs if v[weak] > vec[weak])
    beaten_by = sum(1 for v in train_vecs if _dominates(v, vec))
    if beaten_by:
        dom = f"{beaten_by} training builds beat it on every goal at once"
    else:
        dom = "no training build beats it on every goal at once"
    s2 = (
        f"Its weakest goal is {wn} ({_fmt(full[wn])}, where {direction} is better), "
        f"which {better_on_weak} of {len(train_vecs)} training builds beat on that goal; {dom}."
    )
    return f"{s1} {s2}"

