"""Step 16: justify the recommendation and the rating in plain words.

explain(train_csv) names the recommended player (the top-ranked row from
step1.best) and states, goal by goal, how many training players it beats.
It then lists the input settings that are much more common among the top
players than among all players, each with the counts behind it.

why(train_csv, row) rates one player against the training players, goals
included, and names the goals the player is weakest and strongest on, with
the shares behind each claim.

Every figure is a count or a share computed from the training CSV, so a
skeptical reader can check it by hand. These are associations in the data,
not proof that a setting causes a result.
"""
import math

from step1 import _load, best

_TOP_FRACTION = 0.05   # share of players treated as "top", as in step1.best
_MAX_LEVELS = 8        # numeric inputs with more distinct values are skipped
_MIN_GAP = 0.20        # report a setting if it is this much more common among top players
_MIN_TOP_COUNT = 2     # a setting must appear in at least this many top players
_MAX_REASONS = 6
_FEW_TOP = 10          # warn when fewer top players than this are available


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


def _dominates(a, b):
    """True if a is at least as good as b on every goal and strictly better on one."""
    strict = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strict = True
    return strict


def _name(row):
    """The player's Name cell if usable, else ''."""
    v = row.get("Name")
    return v.strip() if isinstance(v, str) else ""


def _goal_label(c, sg):
    direction = "higher is better" if sg > 0 else "lower is better"
    return f"{c[:-1]} ({direction})"


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes players good.

    Names the recommended player (the top-ranked row from step1.best, or the
    highest mean goal score if step1.best returns nothing), states how many
    training players it beats on each goal, and lists the input settings that
    are at least _MIN_GAP more common among the top players than among all
    players, with the counts needed to recheck each one.
    """
    header, rows = _load(train_csv)
    if not rows:
        return "There are no training players, so no recommendation can be made."
    goals = _goal_spec(header)
    if not goals:
        return ("The file has no goal columns ('+' or '-'), "
                "so players cannot be judged good or bad.")

    vecs = {}
    for i, r in enumerate(rows):
        v = _oriented(r, goals)
        if v is not None:
            vecs[i] = v
    if not vecs:
        return "No player has numeric values for every goal, so nothing can be ranked."

    ids = sorted(vecs)
    n = len(ids)
    lo, hi = _ranges([vecs[i] for i in ids])
    score = {i: sum(_norm(vecs[i], lo, hi)) / len(goals) for i in ids}

    top = [i for i in best(train_csv) if i in vecs]
    if not top:
        k = max(1, int(_TOP_FRACTION * n))
        top = sorted(ids, key=lambda i: (-score[i], i))[:k]
    rec = top[0]
    rec_row = rows[rec]

    def who(i):
        nm = _name(rows[i])
        return f"{nm} (training row {i})" if nm else f"training row {i}"

    lines = [
        f"Recommended player: {who(rec)}, the top-ranked of {n} players "
        f"with complete goal values."
    ]

    lines.append("Goal results of the recommended player:")
    for j, (c, sg) in enumerate(goals):
        x = vecs[rec][j]
        beaten = sum(1 for i in ids if vecs[i][j] < x)
        lines.append(
            f"- {_goal_label(c, sg)}: {_fmt(_finite(rec_row.get(c)))}, better than "
            f"{beaten} of {n} players ({beaten / n:.0%})."
        )

    findings = []
    for pos, c in enumerate(header):
        if not _is_input(c):
            continue
        cell = {i: _level(rows[i].get(c)) for i in ids}
        present = [i for i in ids if cell[i] is not None]
        top_present = [i for i in top if cell[i] is not None]
        if not present or not top_present:
            continue
        levels = {cell[i] for i in present}
        if all(isinstance(v, float) for v in levels) and len(levels) > _MAX_LEVELS:
            continue
        for v in levels:
            n_top = sum(1 for i in top_present if cell[i] == v)
            if n_top < _MIN_TOP_COUNT:
                continue
            n_all = sum(1 for i in present if cell[i] == v)
            p_top = n_top / len(top_present)
            p_all = n_all / len(present)
            if p_top - p_all >= _MIN_GAP:
                findings.append({
                    "gap": p_top - p_all,
                    "pos": pos,
                    "col": c,
                    "val": v,
                    "n_top": n_top,
                    "n_top_all": len(top_present),
                    "p_top": p_top,
                    "n_all": n_all,
                    "n_present": len(present),
                    "p_all": p_all,
                    "has": cell[rec] == v,
                })
    findings.sort(key=lambda f: (-f["gap"], f["pos"], _fmt(f["val"])))

    if findings:
        lines.append(
            "Settings much more common among the top players than among all players:"
        )
        for f in findings[:_MAX_REASONS]:
            status = "has this setting" if f["has"] else "does not have this setting"
            lines.append(
                f"- {f['col']} = {_fmt(f['val'])}: {f['n_top']} of {f['n_top_all']} "
                f"top players ({f['p_top']:.0%}) vs {f['n_all']} of {f['n_present']} "
                f"players ({f['p_all']:.0%}); the recommended player {status}."
            )
    else:
        lines.append(
            "No single setting is much more common among the top players "
            "than among all players."
        )

    if len(top) < _FEW_TOP:
        lines.append(
            f"Only {len(top)} top player(s) were available, so these shares are rough."
        )
    lines.append(
        "These are associations in the training data, not proof that a setting "
        "causes a better result. Each figure above can be recounted from the "
        "training file."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this player (goals included) is good or bad.

    The player's mean normalised goal score (each goal scaled over the training
    players, 0 worst to 1 best) is compared with the training players' scores.
    The player is rated 'good' if at least half the training players score no
    higher, and 'bad' otherwise. The second sentence names the goals the player
    is weakest and strongest on, with the share of training players it beats
    on each.
    """
    header, rows = _load(train_csv)
    goals = _goal_spec(header)
    subject = _name(row) or "This player"
    if not goals:
        return "This file has no goal columns, so the player cannot be rated."

    train = []
    for r in rows:
        v = _oriented(r, goals)
        if v is not None:
            train.append(v)
    if not train:
        return ("No training player has numeric values for every goal, "
                "so the player cannot be rated.")

    rv = _oriented(row, goals)
    if rv is None:
        missing = [c for c, _ in goals if _finite(row.get(c)) is None]
        return (f"{subject} cannot be rated because goal values are missing for: "
                + ", ".join(missing) + ".")

    lo, hi = _ranges(train)
    score = sum(_norm(rv, lo, hi)) / len(goals)
    train_scores = [sum(_norm(v, lo, hi)) / len(goals) for v in train]
    at_most = sum(1 for s in train_scores if s <= score) / len(train_scores)
    verdict = "good" if at_most >= 0.5 else "bad"

    below = []
    for j, (c, _) in enumerate(goals):
        share = sum(1 for v in train if v[j] < rv[j]) / len(train)
        below.append((share, c))
    weak_share, weak_col = min(below, key=lambda t: t[0])
    strong_share, strong_col = max(below, key=lambda t: t[0])

    def describe(c, share):
        raw = _fmt(_finite(row.get(c)))
        return f"{c[:-1]} ({raw}, better than {share:.0%} of training players)"

    beaten_by = sum(1 for v in train if _dominates(v, rv))
    if beaten_by == 0:
        beat_text = "no training player is at least as good on every goal and better on one"
    else:
        beat_text = (f"{beaten_by} of {len(train)} training players are at least as good "
                     "on every goal and better on one")

    sentence1 = (
        f"{subject} is rated {verdict}: on all goals combined, at least as good as "
        f"{at_most:.0%} of the {len(train)} training players "
        f"(score {score:.2f}, where 1 is best on every goal and 0 is worst)."
    )
    sentence2 = (
        f"It is weakest on {describe(weak_col, weak_share)} and strongest on "
        f"{describe(strong_col, strong_share)}, and {beat_text}."
    )
    return sentence1 + " " + sentence2

