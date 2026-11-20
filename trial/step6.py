"""Step 6: justify the recommendation and each rating in plain words.

explain(train_csv) describes the recommended flag set (the top-ranked row from
step1.best) and the settings that are much more common among the top builds
than among all builds. why(train_csv, row) rates one build against the
training builds and says why.

Everything is plain arithmetic on the training CSV. Each claim is a count or
a share that can be checked by hand.
"""
import math

from step1 import _load, best

_TOP_FRACTION = 0.05   # share of builds treated as "top", as in step1.best
_MAX_LEVELS = 8        # numeric inputs with more distinct values are skipped
_MIN_GAP = 0.20        # report a setting if it is this much more common among top builds
_MAX_REASONS = 8


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
    """Goal values with signs applied so higher is always better, or None if any is missing."""
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


def _dominates(a, b):
    """True if a is at least as good as b on every goal and strictly better on one."""
    strict = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strict = True
    return strict


def _level(v):
    """Comparable key for a cell: finite float, non-empty string, or None."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _goal_label(c, sg):
    direction = "higher is better" if sg > 0 else "lower is better"
    return f"{c[:-1]} ({direction})"


def explain(train_csv) -> str:
    """Reasons, in plain words, for what makes builds good.

    Names the recommended build (the top-ranked row from step1.best, or the
    highest mean goal score if step1.best returns nothing), states how it
    does on each goal against the training builds, and lists the settings
    that are at least _MIN_GAP more common among the top builds than among
    all builds. These are associations in the data, not proof of cause.
    """
    header, rows = _load(train_csv)
    if not rows:
        return "There are no training builds, so no recommendation can be made."
    goals = _goal_spec(header)
    if not goals:
        return ("The file has no goal columns ('+' or '-'), "
                "so builds cannot be judged good or bad.")

    vecs = {}
    for i, r in enumerate(rows):
        v = _oriented(r, goals)
        if v is not None:
            vecs[i] = v
    if not vecs:
        return "No build has numeric values for every goal, so nothing can be ranked."

    lo, hi = _ranges(list(vecs.values()))
    score = {i: sum(_norm(v, lo, hi)) / len(goals) for i, v in vecs.items()}

    good = [i for i in best(train_csv) if i in vecs]
    if not good:
        k = max(1, int(_TOP_FRACTION * len(vecs)))
        good = sorted(vecs, key=lambda i: (-score[i], i))[:k]
    rec = good[0]
    rec_row = rows[rec]

    lines = [
        f"Recommended build: training row {rec}, the top-ranked of "
        f"{len(vecs)} builds with complete goal values."
    ]

    on = []
    for c in header:
        if not _is_input(c):
            continue
        v = _level(rec_row.get(c))
        if isinstance(v, float) and v != 0.0:
            on.append(c if v == 1.0 else f"{c}={_fmt(v)}")
        elif isinstance(v, str):
            on.append(f"{c}={v}")
    lines.append(
        "Inputs set in the recommended build: "
        + (", ".join(on) if on else "none")
        + "."
    )

    lines.append("Goal results of the recommended build:")
    for c, sg in goals:
        x = _finite(rec_row.get(c))
        if x is None:
            continue
        all_vals = [_finite(r.get(c)) for r in rows]
        all_vals = [v for v in all_vals if v is not None]
        worse = sum(1 for v in all_vals if sg * v < sg * x)
        lines.append(
            f"- {_goal_label(c, sg)}: {_fmt(x)}, better than "
            f"{worse / len(all_vals):.0%} of the {len(all_vals)} builds that report it."
        )

    base = list(vecs)
    findings = []
    for pos, c in enumerate(header):
        if not _is_input(c):
            continue
        cell = {i: _level(rows[i].get(c)) for i in base}
        present = [i for i in base if cell[i] is not None]
        good_present = [i for i in good if cell[i] is not None]
        if not present or not good_present:
            continue
        levels = {cell[i] for i in present}
        if all(isinstance(v, float) for v in levels) and len(levels) > _MAX_LEVELS:
            continue
        rec_val = _level(rec_row.get(c))
        for v in levels:
            p_all = sum(1 for i in present if cell[i] == v) / len(present)
            p_good = sum(1 for i in good_present if cell[i] == v) / len(good_present)
            if p_good - p_all >= _MIN_GAP:
                findings.append((
                    -(p_good - p_all), pos, v, c, p_good, p_all,
                    len(good_present), len(present), rec_val == v,
                ))

    findings.sort(key=lambda t: (t[0], t[1], _fmt(t[2])))
    if findings:
        lines.append(
            "Settings much more common among the top builds than among all builds:"
        )
        for _, _, v, c, p_good, p_all, ng, na, has in findings[:_MAX_REASONS]:
            status = "has this setting" if has else "does not have this setting"
            lines.append(
                f"- {c} = {_fmt(v)}: in {p_good:.0%} of the {ng} top builds vs "
                f"{p_all:.0%} of the {na} builds; the recommended build {status}."
            )
    else:
        lines.append(
            "No single setting is much more common among the top builds "
            "than among all builds."
        )

    if len(good) < 10:
        lines.append(
            f"Only {len(good)} top build(s) were available, so these shares are rough."
        )
    lines.append(
        "These are associations in the training data, not proof that a setting "
        "causes the difference; confirm the recommended settings with a controlled "
        "build before relying on them."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this build (goals included) is good or bad.

    The build's mean normalised goal score (each goal scaled over the training
    builds, 0 worst to 1 best) is compared with the training builds' scores.
    It is rated 'good' if at least half the training builds score no higher
    than it, and 'bad' otherwise.
    """
    header, rows = _load(train_csv)
    goals = _goal_spec(header)
    if not goals:
        return "This file has no goal columns, so the build cannot be rated."

    train = []
    for r in rows:
        v = _oriented(r, goals)
        if v is not None:
            train.append(v)
    if not train:
        return ("No training build has numeric values for every goal, "
                "so this build cannot be rated.")

    rv = _oriented(row, goals)
    if rv is None:
        missing = [c for c, _ in goals if _finite(row.get(c)) is None]
        return ("This build cannot be rated because its goal values are missing for: "
                + ", ".join(missing) + ".")

    lo, hi = _ranges(train)
    norm = _norm(rv, lo, hi)
    score = sum(norm) / len(goals)
    train_scores = [sum(_norm(v, lo, hi)) / len(goals) for v in train]
    at_most = sum(1 for s in train_scores if s <= score) / len(train_scores)
    verdict = "good" if at_most >= 0.5 else "bad"

    beaten_by = sum(1 for v in train if _dominates(v, rv))
    j_weak = min(range(len(goals)), key=lambda j: norm[j])
    j_strong = max(range(len(goals)), key=lambda j: norm[j])

    def _describe(j):
        c = goals[j][0]
        return f"{c[:-1]} ({_fmt(_finite(row.get(c)))})"

    sentence1 = (
        f"This build is rated {verdict}: its goals are at least as good as "
        f"{at_most:.0%} of the {len(train)} training builds "
        f"(mean normalised goal score {score:.2f}, where 1 is best on every goal "
        f"and 0 is worst)."
    )
    if beaten_by == 0:
        beat_text = "no training build is at least as good on every goal and better on one"
    else:
        beat_text = (f"{beaten_by} training build(s) are at least as good on every goal "
                     f"and better on one")
    sentence2 = (
        f"It is weakest on {_describe(j_weak)} and strongest on {_describe(j_strong)}, "
        f"and {beat_text}."
    )
    return sentence1 + " " + sentence2

