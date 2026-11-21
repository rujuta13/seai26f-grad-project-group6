"""step26.py: justify the high-value group for a campaign, and explain one customer's value.

Standard library only (Python 3.13). Reuses step1 (csv helpers and goal
classification), step2 (cell parsing and the training model), step4 (training
context and row scoring), step5 (input blame), step11 (the scalable Pareto
group, best()) and step12 (calibrated kNN goal estimates for goals a customer
has not yet been measured on).

explain() gives reasons a sceptical sales director can check against the
file. why() gives a value verdict, then the inputs a team can act on, so
the second sentence can be checked against the customer's own records.
"""
from __future__ import annotations

import math
import os
import statistics

from step1 import _classify, _read
from step2 import _model, _num
from step4 import _context, _score
from step5 import blame
from step11 import best
from step12 import guess as _calibrated_guess

_TOP_PROFILE = 5    # input columns listed in explain()
_TOP_ACTION = 3     # actionable input columns named in why()
_MIN_GAP = 0.10     # differences below 10% of the training range are ignored
_ACTIONABLE = ("RECENCY", "MNT", "ACCEPTEDCMP")  # inputs a team can change


def _nums(rows: list[list[str]], j: int) -> list[float]:
    """Numeric values of column j, skipping blank, text and NaN cells."""
    out: list[float] = []
    for r in rows:
        v = _num(r[j])
        if v is not None:
            out.append(v)
    return out


def _is_actionable(name: str) -> bool:
    """True for inputs a team can influence (recency, spend, past offers)."""
    return name.startswith(_ACTIONABLE)


def _profile(header: list[str], rows: list[list[str]],
             group: list[list[str]], inputs: list[int]
             ) -> list[tuple[float, int, str, float, float]]:
    """Inputs where the group's median differs most from all customers."""
    found: list[tuple[float, int, str, float, float]] = []
    for pos, j in enumerate(inputs):
        all_vals = _nums(rows, j)
        grp = _nums(group, j)
        if len(all_vals) < 2 or not grp:
            continue
        span = max(all_vals) - min(all_vals)
        if span <= 0:
            continue
        g = statistics.median(grp)
        a = statistics.median(all_vals)
        gap = (g - a) / span
        if abs(gap) >= _MIN_GAP:
            found.append((abs(gap), pos, header[j], g, a))
    found.sort(key=lambda t: (-t[0], t[1]))
    return found


def _row_gaps(header: list[str], rows: list[list[str]],
              group: list[list[str]], inputs: list[int],
              row: dict) -> list[tuple[str, float, float]]:
    """Inputs where this customer differs most from the typical group member.

    Actionable inputs come first, then the rest, each ordered by gap size.
    """
    found: list[tuple[int, float, int, str, float, float]] = []
    for pos, j in enumerate(inputs):
        name = header[j]
        v = _num(row.get(name))
        if v is None:
            continue
        all_vals = _nums(rows, j)
        grp = _nums(group, j)
        if len(all_vals) < 2 or not grp:
            continue
        span = max(all_vals) - min(all_vals)
        if span <= 0:
            continue
        g = statistics.median(grp)
        gap = (v - g) / span
        if abs(gap) >= _MIN_GAP:
            rank = 0 if _is_actionable(name) else 1
            found.append((rank, -abs(gap), pos, name, v, g))
    found.sort(key=lambda t: (t[0], t[1], t[2]))
    return [(name, v, g) for _, _, _, name, v, g in found]


def explain(train_csv) -> str:
    """Plain-language reasons the high-value group should get the campaign,
    with checks a sceptical reader can repeat."""
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    if not rows or not goals or not inputs:
        return ("Nothing to explain: the file needs customer rows, goal columns "
                "and input columns.")
    picked = best(key)
    if not picked:
        return "Nothing to explain: too few customers to pick a group."

    group = [rows[i] for i in picked]
    points: list[str] = []

    outcomes = ", ".join(
        f"{header[j]} ({'higher' if sign > 0 else 'lower'} is better)"
        for j, sign in goals
    )
    points.append(f"We judge customers on these outcomes: {outcomes}.")
    points.append(
        f"The high-value group is {len(picked)} of {len(rows)} customers. "
        "Each is one that no other customer is at least as good as on every "
        "outcome while being better on at least one. If that set is larger "
        "than 5% of customers, the group keeps those with the highest overall "
        "average. To check: take any customer in the group and confirm that "
        "no other customer is at least as good on every outcome and better "
        "on one."
    )

    for j, _ in goals:
        gv = _nums(group, j)
        av = _nums(rows, j)
        if gv and av:
            points.append(
                f"{header[j]}: the group's median is "
                f"{statistics.median(gv):.4g}, against "
                f"{statistics.median(av):.4g} for all customers."
            )

    for j, _ in goals:
        gv = _nums(group, j)
        av = _nums(rows, j)
        if (header[j].startswith("RESPONSE") and gv and av
                and set(av) <= {0.0, 1.0}):
            points.append(
                "Campaign response: "
                f"{statistics.fmean(gv):.0%} of the group responded to the "
                "campaign recorded in this file, against "
                f"{statistics.fmean(av):.0%} of all customers. (The column is "
                "a 0/1 flag, so its mean is the response rate.)"
            )

    profile = _profile(header, rows, group, inputs)
    if profile:
        lines: list[str] = []
        for _, _, name, g, a in profile[:_TOP_PROFILE]:
            direction = "higher" if g > a else "lower"
            tag = "actionable" if _is_actionable(name) else "background trait"
            lines.append(
                f"   {name} ({tag}): group customers are {direction} "
                f"(median {g:.4g} against {a:.4g} for all customers)."
            )
        points.append(
            "Input values that differ most between the group and all "
            "customers (checkable in the data):\n" + "\n".join(lines)
        )
    else:
        points.append(
            "No input differs by a meaningful amount between the group and "
            "all customers."
        )

    top = blame(key)[:_TOP_PROFILE]
    points.append(
        "Inputs most associated with low value (association, not proof of "
        "cause), strongest first: " + ", ".join(top) + "."
    )
    points.append(
        "Caveat: these patterns come from one file of customers. Check them "
        "on a separate file, and compare the group against a held-back control "
        "group, before committing the campaign budget."
    )

    return "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))


def why(train_csv, row) -> str:
    """Two sentences: the value verdict with outcome scores, then the inputs
    a team can act on, compared with high-value customers.

    `row` is a dict {column name: value}; goal columns may be included. Goals
    not recorded for the customer are estimated from similar customers.
    """
    key = os.fspath(train_csv)
    header, rows = _read(key)
    inputs, goals = _classify(header)
    if not rows or not goals:
        return "Cannot judge: the training file has no customers or no goal columns."

    ctx = _context(key)
    goal_vals = _model(key)["goal_vals"]
    train_means: list[float] = []
    for i in range(len(rows)):
        s = _score([col[i] for col in goal_vals], ctx)
        train_means.append(sum(s) / len(s))
    threshold = statistics.median(train_means)

    names = ctx["goal_names"]
    raw = [_num(row.get(name)) for name in names]
    estimated: list[str] = []
    unscored: list[str] = []
    if any(v is None for v in raw):
        est = _calibrated_guess(key, row)
        for k, name in enumerate(names):
            if raw[k] is not None:
                continue
            e = est.get(name, (math.nan, math.nan, math.nan))[0]
            if math.isfinite(e):
                raw[k] = e
                estimated.append(name)
            else:
                unscored.append(name)

    s = _score(raw, ctx)
    m = sum(s) / len(s)
    verdict = "HIGH VALUE" if m >= threshold else "LOW VALUE"
    strong_name, strong = max(zip(names, s), key=lambda p: p[1])
    weak_name, weak = min(zip(names, s), key=lambda p: p[1])

    first = (
        f"{verdict}: average normalised outcome score {m:.2f} against a typical "
        f"customer's {threshold:.2f} (1 is best on every outcome); weakest on "
        f"{weak_name} ({weak:.2f}), strongest on {strong_name} ({strong:.2f})"
    )
    notes: list[str] = []
    if estimated:
        notes.append(
            f"{', '.join(estimated)} estimated from similar customers because "
            "not recorded"
        )
    if unscored:
        notes.append(
            f"no estimate for {', '.join(unscored)}, so counted as worst"
        )
    if notes:
        first += "; " + "; ".join(notes)
    first += "."

    group_idx = best(key)
    if not group_idx:
        return first + " Too few customers to compare inputs with high-value customers."

    group = [rows[i] for i in group_idx]
    gaps = _row_gaps(header, rows, group, inputs, row)
    acts = [(name, v, g) for name, v, g in gaps if _is_actionable(name)]
    rest = [(name, v, g) for name, v, g in gaps if not _is_actionable(name)]

    def describe(items: list[tuple[str, float, float]]) -> str:
        return "; ".join(
            f"{name} is {v:g} where high-value customers typically have {g:g}"
            for name, v, g in items
        )

    if acts:
        second = "Levers to act on: " + describe(acts[:_TOP_ACTION]) + "."
    elif rest:
        second = (
            "No actionable input differs meaningfully from high-value customers; "
            "the differences are background traits: "
            + describe(rest[:_TOP_ACTION]) + "."
        )
    else:
        second = "No input differs meaningfully from the typical high-value customer."
    return first + " " + second

