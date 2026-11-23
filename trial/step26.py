import numpy as np

from step1 import _load
from step2 import _columns, _as_float, guess
from step4 import _goal_layout, _goal_vec
from step6 import _standardize, _is_num
from step16 import explain as _explain


def explain(train_csv) -> str:
    """Plain-language reasons that separate good rows from the rest, checkable against the data."""
    return _explain(train_csv)


def why(train_csv, row) -> str:
    """One or two sentences on why this row, goals included, is good or bad."""
    G, mu, sd, Z = _standardize(train_csv)
    if len(G) == 0 or G.shape[1] == 0:
        return "Cannot judge this row: the training data has no goal columns."

    maxg, ming = _goal_layout(train_csv)
    names = maxg + ming
    score_all = Z.mean(axis=1)
    median = float(np.median(score_all))

    measured = all(_is_num(row.get(g)) for g in names)
    if measured:
        vec = np.array(_goal_vec(row, maxg, ming), dtype=float)
        raw = {g: float(row[g]) for g in names}
        source = "measured"
    else:
        est = guess(train_csv, row)
        if not all(g in est for g in names):
            return "Cannot judge this row: the goal estimates are incomplete."
        mids = {g: est[g][0] for g in names}
        vec = np.array(_goal_vec(mids, maxg, ming), dtype=float)
        raw = mids
        source = "estimated"

    z = (vec - mu) / sd
    s = float(z.mean())
    verdict = "good" if s >= median else "weak"
    i_min, i_max = int(np.argmin(z)), int(np.argmax(z))

    first = (
        f"This row rates {verdict}: its {source} goals average a standardized score of "
        f"{s:.2f} against a training median of {median:.2f}."
    )
    second = (
        f"Weakest on {names[i_min]} ({raw[names[i_min]]:.4g}, z {z[i_min]:.2f}); "
        f"strongest on {names[i_max]} ({raw[names[i_max]]:.4g}, z {z[i_max]:.2f})"
    )

    # Actionable flags: on/off states where this row differs from most top-5% rows.
    inputs, _ = _columns(train_csv)
    if inputs:
        X = _load(train_csv)["X"]
        top = score_all >= float(np.quantile(score_all, 0.95))
        differing = []
        if top.any():
            for j, c in enumerate(inputs):
                col = X[:, j]
                on_train = col != col.min()
                if not (on_train.any() and (~on_train).any()):
                    continue
                top_share = float(np.mean(on_train[top]))
                row_on = _as_float(row.get(c, 0.0)) != float(col.min())
                if top_share >= 0.5 and not row_on:
                    differing.append(f"{c} is off, but {top_share:.0%} of top-5% rows have it on")
                elif top_share < 0.5 and row_on:
                    differing.append(f"{c} is on, but {1 - top_share:.0%} of top-5% rows have it off")
        if differing:
            second += ("; flags out of line with top rows (associations, not proven causes): "
                       + "; ".join(differing[:3]))

    if not measured:
        second += "; goal values are model estimates, not measurements"
    return first + " " + second + "."

