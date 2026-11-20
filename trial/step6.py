"""step6.py: justify the recommended flag set and rate single builds.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers and
the Pareto-front selection best()), step2 (training model), step4 (goal
scores of one row against the training ranges) and step5 (input blame).

explain() states the rules in plain words and lists checks a sceptical
reader can repeat by hand. why() rates one build from its goal scores.
"""
from __future__ import annotations

import math
import os
import statistics

from step1 import _classify, _read, _to_float, best
from step2 import _model
from step4 import _context, _row_scores, _score
from step5 import blame

_TOP_BLAME = 5
_MIN_ROWS_FOR_FRONT = 20  # best() keeps 5% of rows, so fewer rows give nothing


def _median_of(rows: list[list[str]], j: int) -> float | None:
    vals = [_to_float(r[j]) for r in rows]
    ok = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.median(ok) if ok else None


def explain(train_csv) -> str:
    """Plain-language rules behind the recommended flag set, with checks."""
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    if not rows or not goals or not inputs:
        return ("Nothing to explain: the file needs rows, goal columns "
                "and input columns.")
    picked = best(key)
    if not picked:
        return ("Nothing to explain: too few rows to select good ones "
                f"(need at least {_MIN_ROWS_FOR_FRONT}).")

    ctx = _context(key)
    good = [rows[i] for i in picked]
    lines: list[str] = []

    goal_desc = ", ".join(
        f"{name} ({'maximised' if sign > 0 else 'minimised'})"
        for name, sign in zip(ctx["goal_names"], ctx["signs"])
    )
    lines.append(
        f"1. Goals: {goal_desc}. Each goal is rescaled to 0..1 over the "
        "training range, where 1 is best and 0 is worst."
    )
    lines.append(
        "2. A row is good when no other training row is at least as good on "
        "every goal (the Pareto front). best() keeps at most 5% of the rows "
        f"from that front: {len(picked)} of {len(rows)} here, ranked by "
        "mean score."
    )

    # Recommended set: most common value per input among the good rows.
    rec: dict[int, tuple[str, float]] = {}
    for j in inputs:
        counts: dict[str, int] = {}
        for r in good:
            v = r[j].strip()
            counts[v] = counts.get(v, 0) + 1
        value = min(counts, key=lambda v: (-counts[v], v))
        rec[j] = (value, counts[value] / len(good))

    flat = ", ".join(
        f"{header[j]}={rec[j][0]} ({rec[j][1]:.0%})" for j in inputs
    )
    lines.append(
        "3. Recommended flag set (most common value among good rows, "
        "with its share of those rows):"
    )
    lines.append("   " + flat)

    verbatim = sum(
        1 for r in rows if all(r[j].strip() == rec[j][0] for j in inputs)
    )
    lines.append(
        f"4. That exact combination appears verbatim in {verbatim} of "
        f"{len(rows)} training rows. If it is 0, it was never built in "
        "training and is an untested recombination: treat it as a "
        "hypothesis to measure."
    )

    for j, _ in goals:
        gm = _median_of(good, j)
        am = _median_of(rows, j)
        if gm is not None and am is not None:
            lines.append(
                f"5. Median {header[j]}: {gm:.4g} in good rows against "
                f"{am:.4g} across all rows."
            )

    top = blame(key)[:_TOP_BLAME]
    lines.append(
        "6. Input columns most associated with bad rows (association, not "
        "proof of cause), strongest first: " + ", ".join(top) + "."
    )
    lines.append(
        "7. Caveat: these are patterns in one training file. Check them on "
        "the test file before relying on them."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences: why this row (goals included) is good or bad."""
    key = os.fspath(train_csv)
    header, rows = _read(key)
    _, goals = _classify(header)
    if not rows or not goals:
        return "Cannot judge: the training file has no rows or no goal columns."

    ctx = _context(key)
    model = _model(key)
    goal_vals = model["goal_vals"]
    train = [
        _score([col[i] for col in goal_vals], ctx) for i in range(len(rows))
    ]
    train_means = [sum(s) / len(s) for s in train]
    threshold = statistics.median(train_means)

    names = ctx["goal_names"]
    s = _row_scores(row, ctx)
    m = sum(s) / len(s)
    verdict = "GOOD" if m >= threshold else "BAD"
    missing = [
        name for name in names
        if row.get(name) is None or str(row.get(name)).strip() == ""
    ]

    strongest_name, strongest = max(zip(names, s), key=lambda p: p[1])
    weakest_name, weakest = min(zip(names, s), key=lambda p: p[1])
    beat = sum(1 for t in train if all(a >= b for a, b in zip(t, s)))

    first = (
        f"{verdict}: mean normalised goal score {m:.2f} against a training "
        "median of {:.2f} (1 is best on every goal).".format(threshold)
    )
    second = (
        f"Strongest goal {strongest_name} ({strongest:.2f}); weakest "
        f"{weakest_name} ({weakest:.2f})"
    )
    if missing:
        second += f"; missing goal values score 0 for {', '.join(missing)}"
    second += (
        f"; {beat} training rows are at least as good on every goal."
    )
    return first + " " + second

