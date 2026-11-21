import math

import numpy as np

from step1 import _column_kind, _goal_matrix, _load, best
from step4 import stream
from step5 import blame


def _as_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _goal_names(header):
    return [c for c in header if _column_kind(c) in ("max", "min")]


def _summary(values, is_numeric):
    """Mean for numeric columns, most common label and its share for others."""
    if is_numeric:
        nums = [v for v in values if isinstance(v, float) and math.isfinite(v)]
        if not nums:
            return "no usable values"
        return f"mean {float(np.mean(nums)):.3g}"
    if not values:
        return "no values"
    counts = {}
    for v in values:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1
    label, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return f"most common '{label}' in {count / len(values):.0%} of builds"


def explain(train_csv) -> str:
    header, rows = _load(train_csv)
    if not rows:
        return "No training builds were given, so nothing can be explained."

    goals = _goal_names(header)
    if not goals:
        return "No goal columns (names ending in '+' or '-') were found, so builds cannot be judged."

    good = best(train_csv)
    n = len(rows)
    lines = []
    lines.append(
        "Goals: "
        + ", ".join(
            f"{g} ({'maximise' if _column_kind(g) == 'max' else 'minimise'})"
            for g in goals
        )
        + "."
    )
    lines.append(
        f"Recommended set: {len(good)} of {n} training builds. "
        "A build is in the set if no other build is at least as good on every goal "
        "and strictly better on one; if that front is larger than 5% of the rows, "
        "the best-ranked 5% of it (summed rank over goals) is kept."
    )

    if not good:
        lines.append("No builds qualified, so no flags are recommended from this data.")
    else:
        good_rows = [rows[i] for i in good]
        top_inputs = blame(train_csv)[:3]
        if top_inputs:
            lines.append(
                "Inputs most associated with bad goal values (impurity importance from a "
                "random forest, highest first), compared between recommended and all builds:"
            )
            for c in top_inputs:
                all_vals = [r.get(c) for r in rows]
                good_vals = [r.get(c) for r in good_rows]
                is_numeric = all(isinstance(v, float) for v in all_vals)
                lines.append(
                    f"  - {c}: recommended builds show {_summary(good_vals, is_numeric)}; "
                    f"all builds show {_summary(all_vals, is_numeric)}."
                )

    lines.append(
        f"Caveat: these are patterns in {n} rows only. They do not show that any flag "
        "causes the effect. Confirm the recommended set on held-out builds before relying on it."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    header, rows = _load(train_csv)
    goals = _goal_names(header)
    if not rows or not goals:
        return "Cannot judge this build: no training rows or goal columns."

    values = []
    for g in goals:
        v = _as_float(row.get(g))
        if v is None:
            return f"Cannot judge this build: goal '{g}' is missing or not a finite number."
        values.append(v)

    kept = stream(train_csv, [row])[0]
    verdict = "good" if kept else "bad"
    reason = (
        "no training build is at least as good on every goal and strictly better on one"
        if kept
        else "at least one training build is at least as good on every goal and strictly better on one"
    )

    G = _goal_matrix(header, rows)
    parts = []
    for j, g in enumerate(goals):
        sign = -1.0 if _column_kind(g) == "max" else 1.0
        target = sign * values[j]
        beaten = float((G[:, j] > target).mean())
        parts.append(f"{g}={values[j]:g} beats {beaten:.0%} of training builds")

    return (
        f"Rated {verdict} because {reason}; per goal, " + "; ".join(parts) + "."
    )

