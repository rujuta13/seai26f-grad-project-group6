"""step16.py: justify the shortlist to a manager, and rate one player.

Standard library only (Python 3.13). Reuses step1 (csv helpers and goal
classification), step2 (cell parsing and the training model), step4 (training
context and row scoring), step5 (input blame) and step11 (the scalable
shortlist, best()).

explain() gives reasons a skeptical expert can check against the file.
why() gives a verdict with the goal scores, then the input values where the
player differs most from the typical shortlisted player, so the second
sentence can be checked on video.
"""
from __future__ import annotations

import math
import os
import statistics

from step1 import _classify, _read
from step2 import _model, _num
from step4 import _context, _row_scores, _score
from step5 import blame
from step11 import best

_TOP_PROFILE = 5   # input columns listed in explain()
_TOP_GAPS = 2      # input columns named in why()
_MIN_GAP = 0.10    # differences below 10% of the training range are ignored


def _column(rows: list[list[str]], j: int) -> list[float | None]:
    return [_num(r[j]) for r in rows]


def explain(train_csv) -> str:
    """Plain-language reasons the shortlist is good, with checks to repeat."""
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    if not rows or not goals or not inputs:
        return ("Nothing to explain: the file needs player rows, goal columns "
                "and input columns.")
    picked = best(key)
    if not picked:
        return "Nothing to explain: too few players to pick a shortlist."

    good = [rows[i] for i in picked]
    lines: list[str] = []

    outcomes = ", ".join(
        f"{header[j]} ({'higher' if sign > 0 else 'lower'} is better)"
        for j, sign in goals
    )
    lines.append(f"1. We judge players on these outcomes: {outcomes}.")
    lines.append(
        f"2. The shortlist is the {len(picked)} of {len(rows)} players that no "
        "other player matches or beats on every outcome at once, ranked by "
        "overall average. To check: take any player left out and find a "
        "shortlisted player at least as good on every outcome."
    )

    for j, _ in goals:
        gm_vals = [v for v in _column(good, j) if v is not None]
        am_vals = [v for v in _column(rows, j) if v is not None]
        if gm_vals and am_vals:
            lines.append(
                f"3. {header[j]}: shortlisted players have a median of "
                f"{statistics.median(gm_vals):.4g}, against "
                f"{statistics.median(am_vals):.4g} for all players."
            )

    profile: list[tuple[float, int, str, float, float]] = []
    for pos, j in enumerate(inputs):
        all_vals = [v for v in _column(rows, j) if v is not None]
        good_vals = [v for v in _column(good, j) if v is not None]
        if len(all_vals) < 2 or not good_vals:
            continue
        span = max(all_vals) - min(all_vals)
        if span <= 0:
            continue
        g = statistics.median(good_vals)
        a = statistics.median(all_vals)
        gap = (g - a) / span
        if abs(gap) >= _MIN_GAP:
            profile.append((abs(gap), pos, header[j], g, a))
    profile.sort(key=lambda t: (-t[0], t[1]))

    lines.append(
        "4. Input values that differ most between shortlisted players and all "
        "players (checkable on video or in the attribute screen):"
    )
    if profile:
        for _, _, name, g, a in profile[:_TOP_PROFILE]:
            direction = "higher" if g > a else "lower"
            lines.append(
                f"   {name}: shortlisted players are {direction} "
                f"(median {g:.4g} against {a:.4g} for all players)."
            )
    else:
        lines.append("   No input differs by a meaningful amount.")

    top = blame(key)[:_TOP_PROFILE]
    lines.append(
        "5. Input columns most associated with weaker outcomes (association, "
        "not proof of cause), strongest first: " + ", ".join(top) + "."
    )
    lines.append(
        "6. Caveat: these patterns come from one file of players. Check them "
        "on a separate file and on video before a signing decision."
    )
    return "\n".join(lines)


def _input_gaps(header: list[str], rows: list[list[str]],
                good_rows: list[list[str]], inputs: list[int],
                row: dict) -> list[tuple[str, float, float]]:
    """Inputs where this row differs most from the typical good player."""
    found: list[tuple[float, int, str, float, float]] = []
    for pos, j in enumerate(inputs):
        name = header[j]
        v = _num(row.get(name))
        if v is None:
            continue
        all_vals = [x for x in _column(rows, j) if x is not None]
        good_vals = [x for x in _column(good_rows, j) if x is not None]
        if len(all_vals) < 2 or not good_vals:
            continue
        span = max(all_vals) - min(all_vals)
        if span <= 0:
            continue
        g = statistics.median(good_vals)
        gap = (v - g) / span
        if abs(gap) >= _MIN_GAP:
            found.append((abs(gap), pos, name, v, g))
    found.sort(key=lambda t: (-t[0], t[1]))
    return [(name, v, g) for _, _, name, v, g in found[:_TOP_GAPS]]


def why(train_csv, row) -> str:
    """Two sentences: the verdict with goal scores, then the inputs that
    differ most from a typical good player (checkable on video).

    `row` is a dict {column name: value}; goal columns may be included.
    """
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    if not rows or not goals:
        return "Cannot judge: the training file has no players or no goal columns."

    ctx = _context(key)
    goal_vals = _model(key)["goal_vals"]
    train = [
        _score([col[i] for col in goal_vals], ctx) for i in range(len(rows))
    ]
    threshold = statistics.median([sum(t) / len(t) for t in train])

    s = _row_scores(row, ctx)
    m = sum(s) / len(s)
    verdict = "GOOD" if m >= threshold else "BAD"
    names = ctx["goal_names"]
    strongest_name, strongest = max(zip(names, s), key=lambda p: p[1])
    weakest_name, weakest = min(zip(names, s), key=lambda p: p[1])

    missing = [
        name for name in names
        if row.get(name) is None or str(row.get(name)).strip() == ""
    ]
    first = (
        f"{verdict}: across all outcomes this player scores {m:.2f} against a "
        f"typical training player's {threshold:.2f} (1 is best), strongest on "
        f"{strongest_name} ({strongest:.2f}) and weakest on {weakest_name} "
        f"({weakest:.2f})"
    )
    if missing:
        first += f"; missing values for {', '.join(missing)} count as worst"
    first += "."

    good_idx = best(key)
    if not good_idx:
        second = "Too few players to compare inputs with the shortlisted players."
    else:
        good_rows = [rows[i] for i in good_idx]
        gaps = _input_gaps(header, rows, good_rows, inputs, row)
        if not gaps:
            second = ("No input differs meaningfully from the typical "
                      "shortlisted player.")
        else:
            parts = [
                f"{name} is {v:g} where shortlisted players typically have {g:g}"
                for name, v, g in gaps
            ]
            second = "Largest input differences: " + "; ".join(parts) + "."
    return first + " " + second

