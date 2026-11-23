import math
from collections import Counter

import numpy as np

from step1 import _column_kind, _load, best
from step4 import stream


_MIN_GROUP = 3      # fewest recommended builds that can support a claim
_MIN_EFFECT = 0.5   # smallest standardised gap worth reporting
_TOP = 5            # most input contrasts listed in explain()


def _num(cell):
    """Return cell as a finite float, or None if missing or not numeric."""
    if cell is None or isinstance(cell, bool):
        return None
    try:
        f = float(cell)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def _goals(header):
    return [c for c in header if _column_kind(c) in ("max", "min")]


def _label(goal):
    return goal[:-1].rstrip()


def _direction(goal):
    return "higher is better" if _column_kind(goal) == "max" else "lower is better"


def _signed_matrix(goals, rows):
    """(n, m) matrix where smaller is always better; missing goals are worst."""
    G = np.empty((len(rows), len(goals)), dtype=float)
    for j, g in enumerate(goals):
        sign = -1.0 if _column_kind(g) == "max" else 1.0
        for i, r in enumerate(rows):
            v = _num(r.get(g))
            G[i, j] = math.inf if v is None else sign * v
    return G


def _numeric_contrast(name, all_vals, good_vals):
    a = [x for x in (_num(v) for v in all_vals) if x is not None]
    g = [x for x in (_num(v) for v in good_vals) if x is not None]
    if len(a) < 2 or len(g) < _MIN_GROUP:
        return None
    sd = float(np.std(a))
    if sd == 0.0:
        return None
    mean_a = float(np.mean(a))
    mean_g = float(np.mean(g))
    effect = (mean_g - mean_a) / sd
    text = (
        f"{name}: recommended builds average {mean_g:.3g}, "
        f"against {mean_a:.3g} for all builds"
    )
    return abs(effect), text


def _categorical_contrast(name, all_vals, good_vals):
    a_keys = [str(v) for v in all_vals if not _missing(v)]
    g_keys = [str(v) for v in good_vals if not _missing(v)]
    if not a_keys or len(g_keys) < _MIN_GROUP:
        return None
    count_a = Counter(a_keys)
    count_g = Counter(g_keys)
    best_pick = None
    for lab, ca in count_a.items():
        p = ca / len(a_keys)
        if p <= 0.0 or p >= 1.0:
            continue
        cg = count_g.get(lab, 0)
        if cg < _MIN_GROUP:
            continue
        q = cg / len(g_keys)
        effect = (q - p) / math.sqrt(p * (1.0 - p))
        if best_pick is None or abs(effect) > abs(best_pick[0]):
            best_pick = (effect, lab, p, q)
    if best_pick is None:
        return None
    _, lab, p, q = best_pick
    text = (
        f"{name}: '{lab}' is in {q:.0%} of recommended builds, "
        f"against {p:.0%} of all builds"
    )
    return abs(best_pick[0]), text


def explain(train_csv) -> str:
    """Plain-language reasons, checkable against the data, for what makes rows good."""
    header, rows = _load(train_csv)
    if not rows:
        return "No training builds were given, so there is nothing to explain."
    goals = _goals(header)
    if not goals:
        return "No goal columns (names ending in '+' or '-') were found, so builds cannot be judged."

    good_idx = best(train_csv)
    n = len(rows)
    if not good_idx:
        return "No build qualifies as recommended in this data, so no reasons can be given."

    good_rows = [rows[i] for i in good_idx]
    lines = []
    lines.append(
        f"Recommended builds: {len(good_idx)} of {n} training builds. A build is "
        "recommended if no other build is at least as good on every goal and "
        "strictly better on one; if that set is larger than 5% of the builds, the "
        "best-ranked 5% of it (summed rank over the goals) is kept."
    )

    goal_parts = []
    for g in goals:
        g_vals = [_num(r.get(g)) for r in good_rows]
        a_vals = [_num(r.get(g)) for r in rows]
        g_vals = [v for v in g_vals if v is not None]
        a_vals = [v for v in a_vals if v is not None]
        if g_vals and a_vals:
            goal_parts.append(
                f"{_label(g)} median {float(np.median(g_vals)):.3g} against "
                f"{float(np.median(a_vals)):.3g} for all builds ({_direction(g)})"
            )
    if goal_parts:
        lines.append("On the goals: " + "; ".join(goal_parts) + ".")

    inputs = [c for c in header if _column_kind(c) == "input"]
    contrasts = []
    for c in inputs:
        all_vals = [r.get(c) for r in rows]
        good_vals = [r.get(c) for r in good_rows]
        present = [v for v in all_vals if not _missing(v)]
        numeric = bool(present) and all(_num(v) is not None for v in present)
        if numeric:
            found = _numeric_contrast(c, all_vals, good_vals)
        else:
            found = _categorical_contrast(c, all_vals, good_vals)
        if found is not None and found[0] >= _MIN_EFFECT:
            contrasts.append(found)

    contrasts.sort(key=lambda t: -t[0])
    if contrasts:
        lines.append(
            "Input differences that stand out (at least half a standard deviation, "
            "largest first):"
        )
        for _, text in contrasts[:_TOP]:
            lines.append(f"  - {text}.")
    else:
        lines.append("No input differs clearly between recommended and all builds.")

    lines.append(
        f"Caveat: these are patterns in {n} rows only. They show association, not "
        "cause. Check each point on video or on held-out builds before relying on it."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this build, goals included, is good or bad."""
    header, rows = _load(train_csv)
    goals = _goals(header)
    if not rows or not goals:
        return "Cannot judge this build: no training builds or goal columns were given."

    values = []
    for g in goals:
        v = _num(row.get(g))
        if v is None:
            return (
                f"Cannot judge this build: goal '{_label(g)}' is missing or not a "
                "finite number."
            )
        values.append(v)

    kept = bool(stream(train_csv, [row])[0])

    G = _signed_matrix(goals, rows)
    target = np.array(
        [(-1.0 if _column_kind(g) == "max" else 1.0) * v for g, v in zip(goals, values)],
        dtype=float,
    )
    le = (G <= target).all(axis=1)
    lt = (G < target).any(axis=1)
    dominators = int((le & lt).sum())

    if kept:
        first = (
            "Rated good: no training build is at least as good on every goal and "
            "strictly better on at least one."
        )
    else:
        first = (
            f"Rated bad: {dominators} training build(s) are at least as good on every "
            "goal and strictly better on at least one."
        )

    beaten = [float((G[:, j] > target[j]).mean()) for j in range(len(goals))]
    parts = [
        f"{_label(g)} {v:g} beats {b:.0%}"
        for g, v, b in zip(goals, values, beaten)
    ]
    second = "Share of training builds beaten, per goal: " + "; ".join(parts) + "."
    return f"{first} {second}"

