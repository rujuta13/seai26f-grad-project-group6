import numpy as np

from step1 import _read, _columns, _to_float_column, best
from step4 import stream

_TOP_NUMERIC = 5
_TOP_FLAGS = 5
_MIN_GROUP = 5


def _label(name):
    """Human-readable column name: drop the goal sign suffix."""
    return name[:-1].rstrip() if name[-1:] in ("+", "-") else name


def _is_numeric(rows, col):
    for r in rows:
        try:
            float(r.get(col))
        except (TypeError, ValueError):
            return False
    return True


def _is_binary(vals):
    fin = vals[np.isfinite(vals)]
    return fin.size > 0 and set(np.unique(fin).tolist()) <= {0.0, 1.0}


def explain(train_csv) -> str:
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    if not rows or not goals:
        return ("There are no usable rows or goal columns in the training data, "
                "so no reasons can be given.")

    n = len(rows)
    lines = []

    goal_text = "; ".join(
        ("higher " if sign > 0 else "lower ") + _label(name)
        for name, sign in goals
    )
    lines.append("A row is better when it has " + goal_text + ".")

    front = best(train_csv)
    mask = np.zeros(n, dtype=bool)
    if front:
        mask[np.asarray(front, dtype=int)] = True
    rest = ~mask

    lines.append(
        f"{int(mask.sum())} of {n} rows are standouts: no other row is at least "
        "as good on every goal and strictly better on one. This is a count "
        "from the data, not a judgement of any single row."
    )

    if mask.sum() < _MIN_GROUP or rest.sum() < _MIN_GROUP:
        lines.append("There are too few standout rows to compare them with the rest.")
    else:
        # Numeric inputs whose typical value differs most between standouts and the rest.
        cands = []
        for c in inputs:
            if not _is_numeric(rows, c):
                continue
            vals = _to_float_column(rows, c)
            if _is_binary(vals):
                continue
            fin = np.isfinite(vals)
            a = vals[mask & fin]
            b = vals[rest & fin]
            if len(a) < _MIN_GROUP or len(b) < _MIN_GROUP:
                continue
            sd = float(vals[fin].std())
            if not sd > 0:
                continue
            ma = float(np.median(a))
            mb = float(np.median(b))
            gap = (ma - mb) / sd
            cands.append((abs(gap), c, ma, mb))
        cands.sort(key=lambda t: (-t[0], t[1]))

        if cands:
            lines.append(
                "Numeric inputs where standouts differ most from other rows "
                "(medians; check these on video):"
            )
            for _mag, c, ma, mb in cands[:_TOP_NUMERIC]:
                direction = "higher" if ma > mb else "lower"
                lines.append(
                    f"- {_label(c)}: standouts median {ma:.4g}, other rows "
                    f"median {mb:.4g} ({direction} for standouts)."
                )

        # Binary flags whose share differs most between standouts and the rest.
        flags = []
        for c in inputs:
            if not _is_numeric(rows, c):
                continue
            vals = _to_float_column(rows, c)
            if not _is_binary(vals):
                continue
            fin = np.isfinite(vals)
            on = fin & (vals == 1.0)
            off = fin & (vals == 0.0)
            if on.sum() < _MIN_GROUP or off.sum() < _MIN_GROUP:
                continue
            p_s = float((mask & on).sum()) / float(mask.sum())
            p_r = float((rest & on).sum()) / float(rest.sum())
            flags.append((abs(p_s - p_r), c, p_s, p_r))
        flags.sort(key=lambda t: (-t[0], t[1]))
        flags = [f for f in flags if f[0] > 0][:_TOP_FLAGS]

        if flags:
            lines.append(
                "Yes/no inputs where standouts differ most from other rows "
                "(share of rows with the input on):"
            )
            for _gap, c, p_s, p_r in flags:
                lines.append(
                    f"- {_label(c)}: on for {p_s:.0%} of standouts vs "
                    f"{p_r:.0%} of other rows."
                )

    lines.append(
        "These are patterns in past rows, not proof that any input causes "
        "better results. Check each reason on video, and test a flag with a "
        "controlled comparison before relying on it. The data does not show "
        "injuries, attitude, or anything else not recorded in its columns."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    fieldnames, rows = _read(train_csv)
    _inputs, goals = _columns(fieldnames)
    if not goals or not rows:
        return ("This row cannot be rated: the training data has no usable "
                "goals or rows.")

    vals = {}
    missing = []
    for name, _sign in goals:
        try:
            v = float(row.get(name))
        except (TypeError, ValueError, AttributeError):
            v = float("nan")
        if not np.isfinite(v):
            missing.append(_label(name))
        else:
            vals[name] = v
    if missing:
        return ("This row cannot be rated: its goal values are missing or "
                "non-numeric for " + ", ".join(missing) + ".")

    standout = bool(stream(train_csv, [row])[0])

    beats = []
    details = []
    for name, sign in goals:
        col = _to_float_column(rows, name)
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        v = vals[name]
        if sign > 0:
            frac = float(np.mean(col < v))
        else:
            frac = float(np.mean(col > v))
        beats.append(frac)
        details.append(
            f"{_label(name)} {v:.6g} (training median {float(np.median(col)):.6g})"
        )

    if not beats:
        return "This row cannot be rated: no training goal values are usable."

    mean_beats = float(np.mean(beats))
    if standout and mean_beats >= 0.5:
        verdict = "good"
    elif (not standout) and mean_beats < 0.5:
        verdict = "bad"
    else:
        verdict = "mixed"

    if standout:
        first = "no training row is at least as good on every goal and better on one"
    else:
        first = "at least one training row beats it on every goal"
    sentence1 = (
        f"Rated {verdict}: {first}, and on average it beats "
        f"{100.0 * mean_beats:.0f}% of training rows across the goals."
    )
    sentence2 = (
        "Goal values: " + "; ".join(details)
        + ". This rating reflects only the goal columns in the data, so check "
        "the reasons on video."
    )
    return sentence1 + " " + sentence2

