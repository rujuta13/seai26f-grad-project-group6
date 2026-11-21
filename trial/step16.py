"""Step 16: justify. Plain-language reasons a manager or a video analyst can check.

Serves the client's questions:
  - "Explain to the manager why we should sign this player."
  - "Why did you rate this player low? Give me reasons I can check on video."

Method (builds on steps 1, 3 and 11):
  - The reference group is the top players from step11.best (at most 5% of
    rows), ranked on all goals together. Everyone else is "the rest".
  - Only attributes a scout can check on video are used as reasons. Identifiers
    (num, Name, url), context (Nation, League, Team, Alternative positions),
    composite ratings (OVR, Rank) and goalkeeper-only columns are left out.
  - A numeric attribute gives a reason when the reference group's mean is far
    from the rest's mean (as a fraction of the attribute's range). A text
    attribute gives a reason when one of its values is much more or much less
    common in the reference group.
  - Each goal's difference is checked with a permutation test (step3.same), so
    the manager can see whether the gap could be noise.
  - why() rates one row by its goal shares (as step6.why does) and names the
    attributes where it sits on the wrong side of the reference group.
"""

import math
from collections import Counter

import step1
import step3
import step11

_MIN_GAP = 0.2     # smallest gap that counts as a reason (share difference, or mean
                   # difference as a fraction of the attribute's range)
_MAX_REASONS = 6   # cap on reasons listed by explain()
_MAX_WHY = 3       # cap on attributes named by why()
_MAX_CATS = 40     # text columns with more distinct values than this are not used
_SKIP = frozenset({"num", "Rank", "Name", "url", "Nation", "League", "Team",
                   "Alternative positions", "OVR"})


def _num(v):
    """Finite float for v, or None if the cell is blank, missing or not finite."""
    if isinstance(v, float) and math.isfinite(v):
        return v
    return None


def _key(v):
    """Category label for a cell; blank and missing cells share the key ''."""
    if v is None or v == "" or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return str(v)


def _label(key):
    """Readable category label."""
    return key if key else "(blank)"


def _goal_name(goal):
    """Plain name of a goal column and which direction is better."""
    if goal.endswith("+"):
        return f"{goal[:-1]} (higher is better)"
    return f"{goal[:-1]} (lower is better)"


def _fmt(v):
    """Short, readable number."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _is_numeric(values):
    """True if every non-blank cell is a float (blank cells count as missing)."""
    present = [v for v in values if v is not None and v != ""]
    return bool(present) and all(isinstance(v, float) for v in present)


def _shares(keys):
    """Share of each category label among the given keys."""
    counts = Counter(keys)
    n = len(keys)
    return {k: c / n for k, c in counts.items()}


def _usable(rows, inputs):
    """Checkable input columns as (name, is_numeric) pairs, in header order."""
    keep = []
    for name in inputs:
        if name in _SKIP or name.startswith("GK "):
            continue
        values = [r.get(name) for r in rows]
        if _is_numeric(values):
            keep.append((name, True))
        elif len({_key(v) for v in values}) <= _MAX_CATS:
            keep.append((name, False))
    return keep


def _numeric_facts(name, rows, good, rest):
    """Reference-group facts for a numeric attribute, or None if it is no reason."""
    gv = [x for x in (_num(r.get(name)) for r in good) if x is not None]
    rv = [x for x in (_num(r.get(name)) for r in rest) if x is not None]
    allv = [x for x in (_num(r.get(name)) for r in rows) if x is not None]
    if not gv or not rv or not allv:
        return None
    span = max(allv) - min(allv)
    if span <= 0:
        return None
    mg = sum(gv) / len(gv)
    mr = sum(rv) / len(rv)
    gap = (mg - mr) / span
    if abs(gap) < _MIN_GAP:
        return None
    return {"gap": gap, "good": mg, "rest": mr,
            "binary": set(allv) <= {0.0, 1.0}}


def _numeric_text(name, f):
    """Reason sentence for a numeric attribute (as used by explain)."""
    if f["binary"]:
        if f["gap"] > 0:
            return (f"{name} is on for {f['good']:.0%} of top players versus "
                    f"{f['rest']:.0%} of the rest")
        return (f"{name} is off for {1 - f['good']:.0%} of top players versus "
                f"{1 - f['rest']:.0%} of the rest")
    return (f"{name} averages {_fmt(f['good'])} among top players versus "
            f"{_fmt(f['rest'])} for the rest")


def _biggest_share_gap(sg, sr):
    """The category whose share differs most between the groups: (key, share gap)."""
    best_key, best_gap = None, 0.0
    for k in sorted(set(sg) | set(sr)):
        d = sg.get(k, 0.0) - sr.get(k, 0.0)
        if abs(d) > abs(best_gap):
            best_key, best_gap = k, d
    return best_key, best_gap


def _category_text(name, key, sg, sr):
    """Reason sentence for a text attribute (as used by explain)."""
    return (f"{name} is {_label(key)} for {sg.get(key, 0.0):.0%} of top players "
            f"versus {sr.get(key, 0.0):.0%} of the rest")


def _why_numeric(name, x, facts):
    """Short attribute phrase for a numeric value in why()."""
    if facts["binary"]:
        on = x >= 0.5
        share = facts["good"] if on else 1 - facts["good"]
        state = "on" if on else "off"
        return f"{name} is {state}, which {share:.0%} of top players share"
    return (f"{name} is {_fmt(x)} against a top-player average of "
            f"{_fmt(facts['good'])}")


def explain(train_csv) -> str:
    """Plain-words reasons, checkable by a skeptical expert, for what makes players good.

    The reference group is the top players by step11.best (at most 5% of rows).
    Each checkable attribute is compared between that group and the rest; large
    gaps become reasons, strongest first. Each goal's gap is checked with a
    permutation test (step3.same).
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    goals = goals_max + goals_min
    if not rows or not goals:
        return ("No recommendation: the data needs at least one row and at least one "
                "goal column (names ending in + or -).")
    order = step11.best(train_csv)
    if not order:
        return "No recommendation: no player could be ranked on the goals."

    good_set = set(order)
    good = [rows[i] for i in order]
    rest = [rows[i] for i in range(len(rows)) if i not in good_set]
    if not rest:
        return "No comparison: every training row is in the top group."

    lines = [
        f"Reference group: the {len(good)} best of {len(rows)} players (at most 5%), "
        f"ranked on " + ", ".join(_goal_name(g) for g in goals) + "."
    ]

    reasons = []  # (negative strength, usable-input position, sentence)
    for idx, (name, numeric) in enumerate(_usable(rows, inputs)):
        if numeric:
            facts = _numeric_facts(name, rows, good, rest)
            if facts is not None:
                reasons.append((-abs(facts["gap"]), idx, _numeric_text(name, facts)))
        else:
            sg = _shares([_key(r.get(name)) for r in good])
            sr = _shares([_key(r.get(name)) for r in rest])
            key, gap = _biggest_share_gap(sg, sr)
            if key is not None and abs(gap) >= _MIN_GAP:
                reasons.append((-abs(gap), idx, _category_text(name, key, sg, sr)))

    reasons.sort()
    if reasons:
        lines.append("Reasons the top players differ from the rest, strongest first:")
        lines.extend(f"- {text}." for _, _, text in reasons[:_MAX_REASONS])
    else:
        lines.append(f"No single attribute separates the top players from the rest "
                     f"by at least {_MIN_GAP:.0%}.")

    for g in goals:
        gv = [x for x in (_num(r.get(g)) for r in good) if x is not None]
        rv = [x for x in (_num(r.get(g)) for r in rest) if x is not None]
        if not gv or not rv:
            continue
        verdict = ("the gap is real under a permutation test."
                   if not step3.same(gv, rv) else "the gap could be noise.")
        lines.append(f"{_goal_name(g)}: top players average "
                     f"{_fmt(sum(gv) / len(gv))} versus {_fmt(sum(rv) / len(rv))} "
                     f"for the rest; " + verdict)

    lines.append("These are associations in this data, not proof of cause. To check a "
                 "reason on video, compare the named attribute across the players "
                 "involved; an average gap is a pattern, not a guarantee for one player.")
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this player (goals included) is rated good or bad.

    Per goal, the share of training players the row is at least as good as is
    computed, direction-aware (minimised goals are negated). Missing goal values
    count as worst. The row is rated good if its average share exceeds 0.5. The
    second sentence names the checkable attributes where the row sits on the
    side of the reference group that matches the verdict.
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if not rows or not goals:
        return ("Cannot rate this player: the training data needs at least one row "
                "and at least one goal column.")

    scored = []  # (goal, share of training players this row is at least as good as)
    for g, sign in goals:
        pool = []
        for t in rows:
            x = _num(t.get(g))
            if x is not None:
                pool.append(x * sign)
        if not pool:
            continue
        v = _num(row.get(g))
        share = 0.0 if v is None else sum(1 for p in pool if p <= v * sign) / len(pool)
        scored.append((g, share))
    if not scored:
        return "Cannot rate this player: no training values exist for any goal."

    overall = sum(s for _, s in scored) / len(scored)
    verdict = "good" if overall > 0.5 else "bad"
    favoured = verdict == "good"
    detail = "; ".join(f"{g[:-1]} {s:.0%}" for g, s in scored)
    first = (f"Rated {verdict}: on average it is at least as good as {overall:.0%} of "
             f"training players on the goals (per goal: {detail}).")

    order = step11.best(train_csv)
    good_set = set(order)
    if not order or len(good_set) >= len(rows):
        return first + " No reference group is available to compare its attributes."
    good = [rows[i] for i in order]
    rest = [rows[i] for i in range(len(rows)) if i not in good_set]

    picks = []  # (strength, usable-input position, phrase) on the verdict's side
    for idx, (name, numeric) in enumerate(_usable(rows, inputs)):
        if numeric:
            facts = _numeric_facts(name, rows, good, rest)
            x = _num(row.get(name))
            if facts is None or x is None:
                continue
            mid = (facts["good"] + facts["rest"]) / 2
            if x == mid:
                continue
            toward_good = (x > mid) == (facts["gap"] > 0)
            if toward_good != favoured:
                continue
            picks.append((abs(facts["gap"]), idx, _why_numeric(name, x, facts)))
        else:
            sg = _shares([_key(r.get(name)) for r in good])
            sr = _shares([_key(r.get(name)) for r in rest])
            k = _key(row.get(name))
            d = sg.get(k, 0.0) - sr.get(k, 0.0)
            if abs(d) < _MIN_GAP or (d > 0) != favoured:
                continue
            picks.append((abs(d), idx,
                          f"{name} is {_label(k)}, which {sg.get(k, 0.0):.0%} "
                          f"of top players share"))

    picks.sort(key=lambda p: (-p[0], p[1]))
    texts = [t for _, _, t in picks[:_MAX_WHY]]
    if texts:
        if favoured:
            lead = "Attributes where it sits on the top players' side: "
        else:
            lead = "Attributes where it falls short of the top players: "
        second = lead + "; ".join(texts) + "."
    elif favoured:
        second = ("No attribute clearly places it on the top players' side, so the "
                  "rating rests mainly on its goal values.")
    else:
        second = ("No attribute clearly falls short of the top players, so the rating "
                  "rests mainly on its goal values.")
    return first + " " + second

