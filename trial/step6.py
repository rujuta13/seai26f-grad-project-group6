"""step6.py: plain-language justification of the recommended flag set and of single builds.

Everything statistical or structural is delegated to the earlier steps:
step1 for loading, column classification, the Pareto front and mean goal rank,
step3 for the permutation test, step4 for the keep/discard decision, and step5
for the blamed input columns.
"""
from __future__ import annotations

import numpy as np

from step1 import _classify, _goal_matrix, _load, best
from step3 import same
from step4 import stream
from step5 import blame

_TOP_BLAME = 3   # number of blamed input columns to report in explain()


def _mean_rank(G):
    """Mean rank across goals per row (lower is better). G is a minimisation matrix."""
    n = G.shape[0]
    ranks = np.empty_like(G)
    for k in range(G.shape[1]):
        order = np.argsort(G[:, k], kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        ranks[:, k] = r
    return ranks.mean(axis=1)


def _goal_text(header, cols, row):
    return ", ".join(f"{header[c]}={float(row[header[c]]):g}" for c in cols)


def explain(train_csv) -> str:
    """Plain-words account of what makes rows good and why the recommended set is chosen."""
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")
    maximise, minimise, inputs = _classify(header)
    n = len(rows)
    goal_cols = minimise + maximise
    if not goal_cols:
        return "The file has no goal columns, so no build can be called good."

    G = _goal_matrix(rows, maximise, minimise)
    score = _mean_rank(G)
    rec = int(np.argmin(score))

    good = best(train_csv)
    if not good:
        good = [rec]
    good_set = set(good)
    rest = [i for i in range(n) if i not in good_set]

    lines = []
    lines.append(
        "Definition: a build is 'good' if no other build is at least as good on every "
        "goal and strictly better on at least one (the Pareto front). If the front is "
        "larger than 5% of the builds, the ones with the best mean rank across goals are kept."
    )
    lines.append(
        f"{len(good)} of {n} builds qualify as good. You can check this with step1.best()."
    )

    lines.append(f"Recommended flag set (row {rec}, lowest mean goal rank):")
    for col in inputs:
        lines.append(f"  {header[col]} = {rows[rec][col]}")

    lines.append("Goal values of the recommended build against the other builds:")
    for col in goal_cols:
        name = header[col]
        direction = "lower is better" if col in minimise else "higher is better"
        value = float(rows[rec][col])
        good_med = float(np.median([float(rows[i][col]) for i in good]))
        line = f"  {name} ({direction}): recommended {value:g}, median of good builds {good_med:g}"
        if rest:
            rest_vals = [float(rows[i][col]) for i in rest]
            significant = not same([float(rows[i][col]) for i in good], rest_vals)
            verdict = "a significant gap" if significant else "no significant gap"
            line += f", median of other builds {float(np.median(rest_vals)):g} ({verdict}, permutation test)"
        lines.append(line)

    lines.append(
        f"Input columns most associated with bad builds (step5.blame, top {_TOP_BLAME}):"
    )
    for name in blame(train_csv)[:_TOP_BLAME]:
        col = header.index(name)
        val = rows[rec][col]
        share_good = float(np.mean([rows[i][col] == val for i in good]))
        share_all = float(np.mean([rows[i][col] == val for i in range(n)]))
        lines.append(
            f"  {name} = {val}: present in {share_good:.0%} of good builds "
            f"and {share_all:.0%} of all builds"
        )

    lines.append(
        "Caveats: these are associations observed in one data file, not proof that a "
        "flag causes the difference; flags were not tested in isolation, so interactions "
        "between flags are not separated out."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this build is rated good or bad.

    `row` is a dict {column name: value} that includes the goal columns. An integer
    is also accepted and is read as a data row index of train_csv.
    """
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")
    maximise, minimise, _ = _classify(header)
    goal_cols = minimise + maximise
    if not goal_cols:
        raise ValueError("train_csv has no goal columns")

    if isinstance(row, (int, np.integer)) and not isinstance(row, bool):
        row = dict(zip(header, rows[int(row)]))

    missing = [header[c] for c in goal_cols if header[c] not in row]
    if missing:
        raise ValueError(f"row is missing goal columns: {missing}")

    G = _goal_matrix(rows, maximise, minimise)
    gv = np.array([float(row[header[c]]) for c in minimise]
                  + [-float(row[header[c]]) for c in maximise], dtype=float)

    kept = bool(stream(train_csv, [row])[0])
    goals = _goal_text(header, goal_cols, row)

    if kept:
        beats = [float(np.mean(G[:, k] > gv[k])) for k in range(G.shape[1])]
        pct = 100.0 * float(np.mean(beats))
        return (
            f"Rated good: no training build is at least as good on every goal and strictly "
            f"better on one ({goals}), and this build beats {pct:.0f}% of training builds "
            f"on average across the goals."
        )

    dominators = np.flatnonzero((G <= gv).all(axis=1) & (G < gv).any(axis=1))
    k = int(dominators[0]) if dominators.size else -1
    return (
        f"Rated bad: training build {k} is at least as good on every goal and strictly "
        f"better on at least one, so this build ({goals}) is dominated and adds nothing "
        f"to the Pareto front."
    )

