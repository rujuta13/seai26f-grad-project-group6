"""Step 26: plain-language reasons for what makes rows good, and for one row's rating."""

import numpy as np

from step1 import _read, _columns, _to_float_column, best
from step4 import stream

_TOP_NUMERIC = 5
_TOP_FLAGS = 5
_TOP_REASONS = 3
_MIN_GROUP = 5
_GAP_SD = 0.3      # minimum separation, in SD units, for a row-level numeric reason
_FLAG_GAP = 0.2    # minimum difference in share for a row-level flag reason


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


def _float_or_nan(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _split(train_csv):
    """Training rows, input columns, goals, and a standout mask from step1.best."""
    fieldnames, rows = _read(train_csv)
    inputs, goals = _columns(fieldnames)
    n = len(rows)
    mask = np.zeros(n, dtype=bool)
    front = best(train_csv) if n else []
    if front:
        mask[np.asarray(front, dtype=int)] = True
    return rows, inputs, goals, mask


def _numeric_profile(rows, inputs, mask):
    """Numeric, non-binary inputs: (column, sd, standout median, other median, gap in SD)."""
    rest = ~mask
    out = []
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
        out.append((c, sd, ma, mb, (ma - mb) / sd))
    return out


def _flag_profile(rows, inputs, mask):
    """Binary inputs: (column, share on among standouts, share on among other rows)."""
    rest = ~mask
    out = []
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
        out.append((c, p_s, p_r))
    return out


def explain(train_csv) -> str:
    rows, inputs, goals, mask = _split(train_csv)
    if not rows or not goals:
        return ("There are no usable training rows or goal columns, "
                "so no reasons can be given.")

    n = len(rows)
    k = int(mask.sum())
    rest_n = n - k
    lines = []

    goal_text = "; ".join(
        ("higher " if sign > 0 else "lower ") + _label(name)
        for name, sign in goals
    )
    lines.append("A row is better when it has " + goal_text + ".")
    lines.append(
        f"{k} of {n} rows are standouts: no other row is at least as good on "
        "every goal and strictly better on at least one. A row outside this "
        "set is not automatically poor; it is only beaten on every goal at "
        "once by at least one other row."
    )

    if k < _MIN_GROUP or rest_n < _MIN_GROUP:
        lines.append("There are too few standouts to compare them with the other rows.")
    else:
        numeric = _numeric_profile(rows, inputs, mask)
        numeric.sort(key=lambda t: (-abs(t[4]), t[0]))
        numeric = numeric[:_TOP_NUMERIC]
        if numeric:
            lines.append(
                "Numeric inputs where standouts differ most from other rows "
                "(medians):"
            )
            for c, _sd, ma, mb, _gap in numeric:
                direction = "higher" if ma > mb else "lower"
                lines.append(
                    f"- {_label(c)}: standouts median {ma:.4g}, other rows "
                    f"median {mb:.4g} ({direction} for standouts)."
                )

        flags = _flag_profile(rows, inputs, mask)
        flags = [f for f in flags if abs(f[1] - f[2]) > 0]
        flags.sort(key=lambda t: (-abs(t[1] - t[2]), t[0]))
        flags = flags[:_TOP_FLAGS]
        if flags:
            lines.append(
                "Yes/no inputs where standouts differ most from other rows "
                "(share of rows with the input on):"
            )
            for c, p_s, p_r in flags:
                lines.append(
                    f"- {_label(c)}: on for {p_s:.0%} of standouts vs "
                    f"{p_r:.0%} of other rows."
                )

    lines.append(
        "These are patterns in past rows, not proof that any input causes "
        "better results. Test a suspected cause with a controlled comparison "
        "before relying on it. The data cannot show anything not recorded in "
        "its columns."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    rows, inputs, goals, mask = _split(train_csv)
    if not goals or not rows:
        return ("This row cannot be rated: the training data has no usable "
                "goals or rows.")

    vals = {}
    missing = []
    for name, _sign in goals:
        v = _float_or_nan(row.get(name))
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
        frac = float(np.mean(col < v)) if sign > 0 else float(np.mean(col > v))
        beats.append(frac)
        details.append(
            f"{_label(name)} {v:.6g} (training median {float(np.median(col)):.6g})"
        )

    if not beats:
        return "This row cannot be rated: no training goal values are usable."

    mean_beats = float(np.mean(beats))
    if standout and mean_beats >= 0.5:
        verdict = "high value"
    elif (not standout) and mean_beats < 0.5:
        verdict = "low value"
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
    sentence2 = "Goal values: " + "; ".join(details) + "."

    reasons_text = ""
    k = int(mask.sum())
    rest_n = len(rows) - k
    if verdict != "high value" and k >= _MIN_GROUP and rest_n >= _MIN_GROUP:
        reasons = []
        for c, sd, ma, mb, gap in _numeric_profile(rows, inputs, mask):
            if abs(gap) < _GAP_SD:
                continue
            r = _float_or_nan(row.get(c))
            if not np.isfinite(r):
                continue
            if (gap > 0 and r < ma) or (gap < 0 and r > ma):
                reasons.append((
                    abs(gap),
                    f"{_label(c)} is {r:.4g}, while standouts have a median of "
                    f"{ma:.4g} (other rows {mb:.4g})",
                ))
        for c, p_s, p_r in _flag_profile(rows, inputs, mask):
            if abs(p_s - p_r) < _FLAG_GAP:
                continue
            r = _float_or_nan(row.get(c))
            if r not in (0.0, 1.0):
                continue
            if (p_s > p_r and r == 0.0) or (p_s < p_r and r == 1.0):
                state = "on" if r == 1.0 else "off"
                reasons.append((
                    abs(p_s - p_r),
                    f"{_label(c)} is {state}; it is on for {p_s:.0%} of "
                    f"standouts vs {p_r:.0%} of other rows",
                ))
        reasons.sort(key=lambda t: -t[0])
        reasons = [text for _m, text in reasons[:_TOP_REASONS]]
        if reasons:
            reasons_text = (
                " Inputs that differ from what standouts usually have: "
                + "; ".join(reasons) + "."
            )
        else:
            reasons_text = (
                " No input clearly differs from what standouts usually have."
            )

    caveat = (
        " These are associations in past rows, not proof that changing an "
        "input would change the result; test a change in a controlled way "
        "before acting on it."
    )
    return sentence1 + " " + sentence2 + reasons_text + caveat

