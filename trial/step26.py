import math

import numpy as np

from step1 import _column_kind, _load, best
from step4 import stream
from step16 import (
    _MIN_EFFECT,
    _MIN_GROUP,
    _categorical_contrast,
    _direction,
    _goals,
    _label,
    _missing,
    _num,
    _numeric_contrast,
    _signed_matrix,
)


_TOP_GROUP = 5     # most group-level contrasts listed in explain()
_TOP_CUSTOMER = 3  # most input differences listed in why()


def _share(values, key):
    present = [str(v) for v in values if not _missing(v)]
    if not present:
        return 0.0
    return sum(1 for v in present if v == key) / len(present)


def _customer_gaps(header, rows, good_rows, row):
    """Input values of `row` that differ most from the recommended builds."""
    gaps = []
    for c in header:
        if _column_kind(c) != "input":
            continue
        v = row.get(c)
        if _missing(v):
            continue
        all_vals = [r.get(c) for r in rows]
        good_vals = [r.get(c) for r in good_rows]
        x = _num(v)
        if x is not None:
            a = [u for u in (_num(w) for w in all_vals) if u is not None]
            g = [u for u in (_num(w) for w in good_vals) if u is not None]
            if len(a) < 2 or len(g) < _MIN_GROUP:
                continue
            sd = float(np.std(a))
            if sd == 0.0:
                continue
            mean_g = float(np.mean(g))
            effect = (mean_g - x) / sd
            if abs(effect) < _MIN_EFFECT:
                continue
            if effect > 0:
                note = f"is lower than the recommended average of {mean_g:.3g}"
            else:
                note = f"is higher than the recommended average of {mean_g:.3g}"
            gaps.append(
                (abs(effect), f"{c} is {x:.3g}, which {note}")
            )
        else:
            key = str(v)
            share_g = _share(good_vals, key)
            share_a = _share(all_vals, key)
            n_good = sum(1 for w in good_vals if not _missing(w))
            if n_good < _MIN_GROUP or share_a <= 0.0:
                continue
            diff = share_a - share_g
            if diff < _MIN_EFFECT * 0.1:
                continue
            gaps.append(
                (
                    diff,
                    f"{c} is '{key}', a value found in {share_g:.0%} of "
                    f"recommended builds against {share_a:.0%} of all builds",
                )
            )
    gaps.sort(key=lambda t: -t[0])
    return [text for _, text in gaps[:_TOP_CUSTOMER]]


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes rows good."""
    header, rows = _load(train_csv)
    if not rows:
        return "No past customers were given, so there is nothing to explain."
    goals = _goals(header)
    if not goals:
        return "No goal columns (names ending in '+' or '-') were found, so customers cannot be judged."

    good_idx = best(train_csv)
    n = len(rows)
    if not good_idx:
        return "No customer qualifies as recommended in this data, so no reasons can be given."

    good_rows = [rows[i] for i in good_idx]
    lines = [
        f"The recommended group is {len(good_idx)} of {n} past customers. A customer "
        "is in the group if no other customer is at least as good on every goal and "
        "strictly better on one; if that set is larger than 5% of customers, the "
        "best-ranked 5% of it (summed rank over the goals) is kept."
    ]

    goal_parts = []
    for g in goals:
        g_vals = [v for v in (_num(r.get(g)) for r in good_rows) if v is not None]
        a_vals = [v for v in (_num(r.get(g)) for r in rows) if v is not None]
        if g_vals and a_vals:
            goal_parts.append(
                f"{_label(g)} median {float(np.median(g_vals)):.3g} against "
                f"{float(np.median(a_vals)):.3g} for all customers "
                f"({_direction(g)})"
            )
    if goal_parts:
        lines.append("On the goals: " + "; ".join(goal_parts) + ".")

    contrasts = []
    for c in (c for c in header if _column_kind(c) == "input"):
        all_vals = [r.get(c) for r in rows]
        good_vals = [r.get(c) for r in good_rows]
        present = [v for v in all_vals if not _missing(v)]
        numeric = bool(present) and all(_num(v) is not None for v in present)
        found = (
            _numeric_contrast(c, all_vals, good_vals)
            if numeric
            else _categorical_contrast(c, all_vals, good_vals)
        )
        if found is not None and found[0] >= _MIN_EFFECT:
            contrasts.append(found)

    contrasts.sort(key=lambda t: -t[0])
    if contrasts:
        lines.append(
            "Customer traits that stand out in the recommended group (at least half "
            "a standard deviation, largest first):"
        )
        for _, text in contrasts[:_TOP_GROUP]:
            lines.append(f"  - {text}.")
    else:
        lines.append("No customer trait differs clearly between the recommended group and all customers.")

    lines.append(
        f"Caveat: these are patterns in {n} past customers. They show association, "
        "not cause. Before committing the full campaign budget, test the group against "
        "a held-out or randomly chosen comparison group."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this row, goals included, is good or bad."""
    header, rows = _load(train_csv)
    goals = _goals(header)
    if not rows or not goals:
        return "Cannot judge this customer: no past customers or goal columns were given."

    values = []
    for g in goals:
        v = _num(row.get(g))
        if v is None:
            return (
                f"Cannot judge this customer: goal '{_label(g)}' is missing or not a "
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
            "Rated good: no past customer is at least as good on every goal and "
            "strictly better on at least one."
        )
    else:
        first = (
            f"Rated low value: {dominators} past customer(s) are at least as good on "
            "every goal and strictly better on at least one."
        )

    beaten = [float((G[:, j] > target[j]).mean()) for j in range(len(goals))]
    parts = [f"{_label(g)} {v:g} beats {b:.0%} of past customers"
             for g, v, b in zip(goals, values, beaten)]
    second = "Share of past customers beaten, per goal: " + "; ".join(parts) + "."

    actions = []
    if not kept:
        good_idx = best(train_csv)
        if good_idx:
            good_rows = [rows[i] for i in good_idx]
            gaps = _customer_gaps(header, rows, good_rows, row)
            if gaps:
                actions = [
                    "Traits that differ from the recommended group, which may be "
                    "worth checking: " + "; ".join(gaps) + "."
                ]

    return " ".join([first, second] + actions)

