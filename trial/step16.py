"""step16.py: plain-language justification for the client's two questions.

explain() says why the good players are good, using only traits and goals that can
be checked on video. why() gives one verdict sentence and one trait sentence for a
single player.

Reused from earlier steps: loading, classification, goal matrices and value coercion
from step1; the good-player group (Pareto front capped at 5%) from step11.best; the
mean-rank helper from step6; the permutation test from step3; the keep/discard
decision from step4.stream; and the identifier-free trait filter from step11.
"""
from __future__ import annotations

import numpy as np

from step1 import _classify, _goal_matrix, _load, _to_value
from step3 import same
from step4 import stream
from step6 import _mean_rank
from step11 import _trait_inputs, best

_TOP_TRAITS = 5     # traits named in explain()
_MIN_GAP_SD = 0.3   # a trait must differ by at least this many sd between groups
_WHY_TRAITS = 2     # traits named in why()


def _good_rows(train_csv, rows, G):
    """Row indices of the good players: the Pareto front capped at 5%, or the best single row."""
    idx = best(train_csv)
    if not idx:
        idx = [int(np.argmin(_mean_rank(G)))]
    return sorted(int(i) for i in idx)


def _profile(header, rows, inputs, good_mask):
    """Numeric traits with their medians for good and other players, and the gap in sd units."""
    profile = []
    for c in _trait_inputs(header, inputs):
        vals = [_to_value(r[c]) for r in rows]
        if not all(isinstance(v, float) for v in vals):
            continue
        v = np.array(vals, dtype=float)
        sd = float(v.std())
        if sd == 0.0 or not np.isfinite(sd):
            continue
        g = v[good_mask]
        r = v[~good_mask]
        good_med = float(np.median(g))
        rest_med = float(np.median(r)) if r.size else float("nan")
        gap = (good_med - rest_med) / sd if r.size else 0.0
        profile.append({
            "col": c,
            "name": header[c],
            "good": good_med,
            "rest": rest_med,
            "gap": gap,
            "sd": sd,
            "values": v,
        })
    return profile


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes players good."""
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")
    maximise, minimise, inputs = _classify(header)
    goal_cols = minimise + maximise
    if not goal_cols:
        return "The file has no goal columns, so no player can be called good."

    n = len(rows)
    G = _goal_matrix(rows, maximise, minimise)
    good_idx = _good_rows(train_csv, rows, G)
    good_mask = np.zeros(n, dtype=bool)
    good_mask[good_idx] = True

    lines = [
        "Definition: a player is 'good' if no other player is at least as good on every "
        "goal and strictly better on at least one (the Pareto front). If the front is "
        "larger than 5% of the players, the ones with the best mean rank across goals are kept.",
        f"{int(good_mask.sum())} of {n} players qualify as good.",
    ]

    if good_mask.all():
        lines.append("Every player is in the good group, so no comparison with other players is possible.")
        lines.append(_caveat())
        return "\n".join(lines)

    lines.append("Goal values, good players against the other players:")
    for col in goal_cols:
        name = header[col]
        direction = "lower is better" if col in minimise else "higher is better"
        vals = np.array([float(r[col]) for r in rows], dtype=float)
        g = vals[good_mask]
        r = vals[~good_mask]
        verdict = "a significant gap" if not same(g, r) else "no significant gap"
        lines.append(
            f"  {name} ({direction}): good players median {float(np.median(g)):g}, "
            f"other players median {float(np.median(r)):g} ({verdict}, permutation test)"
        )

    profile = _profile(header, rows, inputs, good_mask)
    candidates = sorted(
        (p for p in profile if abs(p["gap"]) >= _MIN_GAP_SD),
        key=lambda p: (-abs(p["gap"]), p["col"]),
    )
    chosen = []
    for p in candidates:
        if len(chosen) >= _TOP_TRAITS:
            break
        if not same(p["values"][good_mask], p["values"][~good_mask]):
            chosen.append(p)

    if chosen:
        lines.append("Traits that set good players apart (each can be checked on video):")
        for p in chosen:
            direction = "higher" if p["gap"] > 0 else "lower"
            lines.append(
                f"  {p['name']}: good players are {direction}, median {p['good']:g} "
                f"against {p['rest']:g} for the other players"
            )
    else:
        lines.append(
            "No trait separates good players from the rest by a significant, sizeable gap."
        )

    lines.append(_caveat())
    return "\n".join(lines)


def _caveat():
    return (
        "Caveats: these are associations observed in one data file, not proof that a trait "
        "causes a player to be good. Check each named trait on video before relying on it, "
        "and treat the goal values as the file's own measurements."
    )


def why(train_csv, row) -> str:
    """One or two sentences on why this row (goals included) is good or bad.

    `row` is a dict {column name: value} that includes the goal columns. An integer
    is also accepted and is read as a data row index of train_csv.
    """
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")
    maximise, minimise, inputs = _classify(header)
    goal_cols = minimise + maximise
    if not goal_cols:
        raise ValueError("train_csv has no goal columns")

    if isinstance(row, (int, np.integer)) and not isinstance(row, bool):
        row = dict(zip(header, rows[int(row)]))

    missing = [header[c] for c in goal_cols if header[c] not in row]
    if missing:
        raise ValueError(f"row is missing goal columns: {missing}")

    n = len(rows)
    G = _goal_matrix(rows, maximise, minimise)
    gv = np.array(
        [float(row[header[c]]) for c in minimise]
        + [-float(row[header[c]]) for c in maximise],
        dtype=float,
    )
    goals = ", ".join(f"{header[c]}={float(row[header[c]]):g}" for c in goal_cols)

    good_idx = _good_rows(train_csv, rows, G)
    good_mask = np.zeros(n, dtype=bool)
    good_mask[good_idx] = True
    profile = _profile(header, rows, inputs, good_mask)

    # Trait comparison of this player with the good players.
    traits_vs_good = []
    for p in profile:
        rv = _to_value(row.get(p["name"]))
        if isinstance(rv, float):
            traits_vs_good.append((p, (rv - p["good"]) / p["sd"], rv))

    kept = bool(stream(train_csv, [row])[0])

    if kept:
        s1 = (
            f"Rated good: no training player is at least as good on every goal and strictly "
            f"better on one ({goals}), so this player is on the Pareto front."
        )
        strong = sorted(
            (t for t in traits_vs_good if t[1] >= _MIN_GAP_SD),
            key=lambda t: (-t[1], t[0]["col"]),
        )[:_WHY_TRAITS]
        if strong:
            s2 = "Checkable strengths against the good players: " + ", ".join(
                f"{p['name']} {rv:g} (good players' median {p['good']:g})"
                for p, _, rv in strong
            ) + "."
        else:
            s2 = "No single trait stands out above the good players' medians."
        return f"{s1} {s2}"

    dominators = np.flatnonzero((G <= gv).all(axis=1) & (G < gv).any(axis=1))
    k = int(dominators[0]) if dominators.size else -1
    s1 = (
        f"Rated low: training player {k} is at least as good on every goal and strictly "
        f"better on at least one, so this player ({goals}) adds nothing to the Pareto front."
    )
    weak = sorted(
        (t for t in traits_vs_good if t[1] <= -_MIN_GAP_SD),
        key=lambda t: (t[1], t[0]["col"]),
    )[:_WHY_TRAITS]
    if weak:
        s2 = "Checkable weaknesses against the good players: " + ", ".join(
            f"{p['name']} {rv:g} (good players' median {p['good']:g})"
            for p, _, rv in weak
        ) + "."
    else:
        s2 = "No single trait falls clearly below the good players' medians."
    return f"{s1} {s2}"

