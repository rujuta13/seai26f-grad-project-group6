import csv
import math

import numpy as np

from step1 import best as _best


def _load(train_csv):
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _split_columns(cols):
    goal_max = [c for c in cols if c.endswith("+")]
    goal_min = [c for c in cols if c.endswith("-")]
    inputs = [c for c in cols if not c.endswith(("+", "-", "X"))]
    return inputs, goal_max, goal_min


def _average_ranks(values):
    """Ranks 1..N with ties replaced by their average rank."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts + 1) / 2.0
    return avg[inverse]


def _row_goal_vector(row, goal_min, goal_max):
    """All goals as minimisation values; missing or non-finite = worst (+inf)."""
    vals = []
    for c in goal_min:
        v = _to_float(row.get(c))
        vals.append(v if math.isfinite(v) else math.inf)
    for c in goal_max:
        v = _to_float(row.get(c))
        vals.append(-v if math.isfinite(v) else math.inf)
    return vals


def _goal_matrix(goal_min, goal_max, rows):
    goals = goal_min + goal_max
    G = np.empty((len(rows), len(goals)), dtype=float)
    for j, c in enumerate(goals):
        v = np.array([_to_float(r.get(c)) for r in rows], dtype=float)
        missing = ~np.isfinite(v)
        if c.endswith("+"):
            v = -v
        v[missing] = np.inf
        G[:, j] = v
    return G


def _badness(G):
    """Mean normalised rank across goals: 0 = best, 1 = worst."""
    n = G.shape[0]
    b = np.zeros(n, dtype=float)
    for j in range(G.shape[1]):
        b += (_average_ranks(G[:, j]) - 1.0) / max(n - 1, 1)
    return b / G.shape[1]


def _pareto_size(G):
    n = G.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        weakly = np.all(G <= G[i], axis=1)
        strictly = np.any(G < G[i], axis=1)
        dominated[i] = np.any(weakly & strictly)
    return int((~dominated).sum())


def _input_values(rows, c):
    vals = np.array([_to_float(r.get(c)) for r in rows], dtype=float)
    return np.where(np.isnan(vals), 0.0, vals)


def explain(train_csv) -> str:
    """Plain-language reasons for what makes builds good, checkable by a skeptic."""
    cols, rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max
    n = len(rows)

    if n == 0 or not goals:
        return "No usable training rows or goal columns; nothing can be explained."

    G = _goal_matrix(goal_min, goal_max, rows)
    badness = _badness(G)

    lines = [
        f"Data: {n} builds, {len(inputs)} input flags, goals: {', '.join(goals)}.",
        "Each build is scored by its average normalised rank across all goals "
        "(better goal values rank higher; missing values count as worst).",
    ]

    if np.all(badness == badness[0]):
        lines.append("All builds score the same on the goals, so no flag can be "
                     "blamed or credited.")
        return "\n".join(lines)

    q75 = float(np.quantile(badness, 0.75))
    q25 = float(np.quantile(badness, 0.25))
    bad = badness >= q75
    good = (~bad) & (badness <= q25)
    if not good.any():
        good = ~bad
    if not bad.any() or not good.any():
        lines.append("The goals do not split the builds into clearly good and bad "
                     "groups, so no flag explanation is available.")
        return "\n".join(lines)

    lines.append(f"Bad group (top quartile of badness): {int(bad.sum())} builds. "
                 f"Good group (bottom quartile): {int(good.sum())} builds.")
    lines.append(f"Pareto-optimal builds (no other build is at least as good on every "
                 f"goal and better on one): {_pareto_size(G)} of {n}.")

    # Goal medians in each group, raw values, for direct checking.
    lines.append("Median goal values, good group vs bad group:")
    for g in goals:
        raw = np.array([_to_float(r.get(g)) for r in rows], dtype=float)
        mg = np.nanmedian(raw[good]) if np.isfinite(raw[good]).any() else math.nan
        mb = np.nanmedian(raw[bad]) if np.isfinite(raw[bad]).any() else math.nan
        lines.append(f"  {g}: good {mg:.4g}, bad {mb:.4g}")

    # Which input flags separate the groups.
    rank_bad = _average_ranks(badness)
    scored = []
    for c in inputs:
        vals = _input_values(rows, c)
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        if len(uniq) == 2:
            on = vals == uniq[-1]
            on_bad = float(on[bad].mean())
            on_good = float(on[good].mean())
            score = on_bad - on_good
            if score == 0:
                continue
            direction = "worse" if score > 0 else "better"
            text = (f"- {c}: on in {on_bad:.0%} of bad builds vs {on_good:.0%} of "
                    f"good builds; 'on' goes with {direction} goals.")
            scored.append((abs(score), text))
        else:
            rv = _average_ranks(vals)
            if rv.std() == 0 or rank_bad.std() == 0:
                continue
            rho = float(np.corrcoef(rv, rank_bad)[0, 1])
            if not math.isfinite(rho) or rho == 0:
                continue
            direction = "worse" if rho > 0 else "better"
            text = (f"- {c}: higher values go with {direction} goals "
                    f"(rank correlation {rho:+.2f}).")
            scored.append((abs(rho), text))

    scored.sort(key=lambda t: -t[0])
    if scored:
        lines.append("Strongest flag associations (largest first):")
        lines.extend(t for _, t in scored[:5])
    else:
        lines.append("No input flag separates the good and bad groups.")

    # The recommended set: Pareto-best builds, summarised per flag.
    rec = _best(train_csv)
    if rec:
        lines.append(f"Recommended builds: {len(rec)} Pareto-best rows.")
        on_flags, off_flags = [], []
        for c in inputs:
            vals = _input_values(rows, c)
            uniq = np.unique(vals)
            if len(uniq) != 2:
                continue
            on = vals == uniq[-1]
            frac = float(on[rec].mean())
            if frac >= 0.9:
                on_flags.append(c)
            elif frac <= 0.1:
                off_flags.append(c)
        lines.append("  Flags on in at least 90% of recommended builds: "
                     + (", ".join(on_flags) if on_flags else "none") + ".")
        lines.append("  Flags off in at least 90% of recommended builds: "
                     + (", ".join(off_flags) if off_flags else "none") + ".")

    lines.append("Caveat: these are associations in one dataset, not proof that "
                 "switching a flag causes the change. Verify any claim by building "
                 "and measuring the flag set directly.")
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences saying why this build is rated good or bad."""
    cols, rows = _load(train_csv)
    inputs, goal_max, goal_min = _split_columns(cols)
    goals = goal_min + goal_max
    n = len(rows)

    if n == 0 or not goals:
        return "Cannot rate this build: there are no usable training rows or goals."

    G = _goal_matrix(goal_min, goal_max, rows)
    v = np.array(_row_goal_vector(row, goal_min, goal_max), dtype=float)

    # Per goal: fraction of training builds that beat this build (ties count half).
    frac_better = np.zeros(len(goals), dtype=float)
    for j in range(len(goals)):
        lt = int(np.sum(G[:, j] < v[j]))
        eq = int(np.sum(G[:, j] == v[j]))
        frac_better[j] = (lt + 0.5 * eq) / n
    p = float(frac_better.mean())

    if p <= 0.25:
        verdict = "good"
    elif p >= 0.75:
        verdict = "bad"
    else:
        verdict = "middling"

    worst_j = int(np.argmax(frac_better))
    worst_goal = goals[worst_j]

    dominators = int(np.sum(np.all(G <= v, axis=1) & np.any(G < v, axis=1)))
    if dominators > 0:
        dom_text = (f"it is beaten by {dominators} training build(s) that are at least "
                    f"as good on every goal and better on one")
    else:
        dom_text = "no training build beats it on every goal"

    sentence1 = (f"This build is {verdict}: on average {p:.0%} of training builds do "
                 f"better on each goal, and {dom_text}.")
    sentence2 = (f"Its weakest goal is {worst_goal}, where "
                 f"{frac_better[worst_j]:.0%} of training builds do better.")
    return f"{sentence1} {sentence2}"

