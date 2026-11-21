import numpy as np

from step1 import _read, _columns, _to_float_column, best
from step4 import stream
from step5 import blame

_TOP_FLAGS = 5
_MIN_SIDE = 5


def _direction(sign: float) -> str:
    return "maximise" if sign > 0 else "minimise"


def explain(train_csv) -> str:
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    if not rows or not goals:
        return ("There are no usable training rows or goal columns, "
                "so no reasons can be given.")

    lines = []
    lines.append(
        "Goals: "
        + "; ".join(f"{name} ({_direction(sign)})" for name, sign in goals)
        + "."
    )

    front = best(train_csv)
    lines.append(
        f"The recommended set has {len(front)} of {len(rows)} training builds. "
        "These are builds that no other training build beats on every goal "
        "at once, trimmed to at most 5% of rows by a simple sum of "
        "min-max normalised goal values. The trimming uses that sum, "
        "not a fitted model."
    )

    ordering = blame(train_csv)
    lines.append(
        "Binary inputs ranked by how strongly they are associated with worse "
        "goals (association in the observed builds, not proof of cause):"
    )

    shown = 0
    for c in ordering:
        if shown >= _TOP_FLAGS:
            break
        vals = _to_float_column(rows, c)
        if not set(np.unique(vals).tolist()) <= {0.0, 1.0}:
            continue
        on = vals == 1.0
        off = ~on
        if on.sum() < _MIN_SIDE or off.sum() < _MIN_SIDE:
            continue
        shown += 1
        parts = []
        for name, _sign in goals:
            y = _to_float_column(rows, name)
            yo = y[on & np.isfinite(y)]
            yf = y[off & np.isfinite(y)]
            if len(yo) == 0 or len(yf) == 0:
                continue
            parts.append(
                f"{name} averages {yf.mean():.4g} when off vs {yo.mean():.4g} when on"
            )
        lines.append(
            f"- {c}: on in {int(on.sum())} builds, off in {int(off.sum())}; "
            + ", ".join(parts)
            + "."
        )

    if shown == 0:
        lines.append("- No binary input had enough builds on both sides to compare.")

    lines.append(
        "Non-binary inputs are ranked by the blame order but not described here. "
        "Before trusting any flag, check it with a controlled pair of builds "
        "that differ only in that flag."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    fieldnames, rows = _read(train_csv)
    _inputs, goals = _columns(fieldnames)
    if not goals or not rows:
        return ("This build cannot be rated: the training data has no usable "
                "goals or rows.")

    vals = {}
    missing = []
    for name, _sign in goals:
        try:
            v = float(row.get(name))
        except (TypeError, ValueError, AttributeError):
            v = float("nan")
        if not np.isfinite(v):
            missing.append(name)
        else:
            vals[name] = v
    if missing:
        return ("This build cannot be rated: its goal values are missing or "
                "non-numeric for " + ", ".join(missing) + ".")

    not_dominated = bool(stream(train_csv, [row])[0])

    pct = []
    details = []
    for name, sign in goals:
        col = _to_float_column(rows, name)
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        v = vals[name]
        if sign > 0:
            beaten = float(np.mean(col < v))
        else:
            beaten = float(np.mean(col > v))
        pct.append(beaten)
        details.append(
            f"{name} = {v:.6g} (training median {float(np.median(col)):.6g})"
        )

    if not pct:
        return "This build cannot be rated: no training goal values are usable."

    mean_pct = float(np.mean(pct))
    if not_dominated and mean_pct >= 0.5:
        verdict = "good"
    elif (not not_dominated) and mean_pct < 0.5:
        verdict = "bad"
    else:
        verdict = "mixed"

    if not_dominated:
        first = "no training build beats it on every goal at once"
    else:
        first = "at least one training build beats it on every goal"
    sentence1 = (
        f"Rated {verdict}: {first}, and on average it beats "
        f"{100.0 * mean_pct:.0f}% of training builds across the goals."
    )
    sentence2 = "Goal values: " + "; ".join(details) + "."
    return sentence1 + " " + sentence2

