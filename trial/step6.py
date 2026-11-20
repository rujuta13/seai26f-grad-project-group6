"""Step 6: justify.

explain(): plain-language reasons for the recommended flag set. Each reason is a
count or comparison a sceptical engineer can recompute from the training csv.

why(): one or two sentences on why a single build is rated good or bad.

Shared definitions:
  - A "good" build is one that step1.best returns: on the Pareto front of the
    goals, capped at 5% of rows. No other build is at least as good on every
    goal and better on at least one.
  - Goal score: each goal is scaled to [0, 1] over the training builds (0 = worst
    training value, 1 = best), then averaged over goals. This is the same scale
    step1.best uses to rank the front.
"""
import os
from collections import Counter

from step1 import _load, best
from step3 import same

MIN_SUPPORT = 3    # a setting must appear at least this often on both sides of a comparison
TOP_SETTINGS = 5   # how many associated settings explain() reports
MAJORITY = 0.6     # a recommended setting is flagged "mixed" below this share of good builds
_CACHE = {}


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _oriented(row, goals):
    """Goal vector with higher always better, or None if any goal cell is unusable."""
    out = []
    for c, sign in goals:
        try:
            out.append(sign * float(row[c]))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def _dominates(q, p):
    """True if q is at least as good as p on every goal and better on at least one."""
    return all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p))


def _scale(vectors, g):
    lo = [min(v[j] for v in vectors) for j in range(g)]
    hi = [max(v[j] for v in vectors) for j in range(g)]
    return lo, hi


def _score(o, lo, hi):
    """Mean of the per-goal scaled values; a constant goal contributes 0, as in step1."""
    total = 0.0
    for j in range(len(o)):
        span = hi[j] - lo[j]
        if span > 0:
            total += (o[j] - lo[j]) / span
    return total / len(o)


def _data(train_csv):
    """(rows, inputs, goals, good) for a training file; good is a set of row indices."""
    stamp = (train_csv, os.path.getmtime(train_csv))
    hit = _CACHE.get(stamp)
    if hit is None:
        rows, inputs, goals = _load(train_csv)
        good = set(best(train_csv))
        hit = (rows, inputs, goals, good)
        _CACHE.clear()
        _CACHE[stamp] = hit
    return hit


def _majority(rows, inputs, idx):
    """Per input column: (most common value among rows idx, how many rows in idx have it)."""
    out = {}
    for c in inputs:
        counts = Counter(_text(rows[i].get(c)) for i in idx)
        value, k = counts.most_common(1)[0]
        out[c] = (value, k)
    return out


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes rows good."""
    rows, inputs, goals, good_set = _data(train_csv)
    n = len(rows)
    if n == 0:
        return "There are no training rows, so no flag set can be recommended."
    if not goals:
        return ("The file has no goal columns (names ending in + or -), so no build "
                "can be called good and no set can be recommended.")
    if not inputs:
        return "The file has goals but no input settings, so there is nothing to recommend."

    oriented = [_oriented(r, goals) for r in rows]
    usable = [i for i, o in enumerate(oriented) if o is not None]
    if not usable:
        return "No row has numeric values for every goal, so builds cannot be compared."
    lo, hi = _scale([oriented[i] for i in usable], len(goals))
    score = {i: _score(oriented[i], lo, hi) for i in usable}

    good = sorted(i for i in good_set if i in score)
    if not good:
        return ("No build is unusually good on all goals together, "
                "so there is no evidence-based recommended set.")

    goal_names = ", ".join(c for c, _ in goals)
    lines = [
        f"Goals: {goal_names}. Each goal is scaled to 0-1 over the {len(usable)} training "
        "builds with complete goal values (0 = worst, 1 = best). A build's goal score is "
        "the average of its scaled goals.",
        f"Good builds: {len(good)} of {n} rows, the Pareto front of the goals (no other build "
        "is at least as good on every goal and better on one), capped at 5% of rows. The "
        "recommendation below is drawn from these rows.",
        "Recommended settings (most common value among the good builds):",
    ]

    maj = _majority(rows, inputs, good)
    for c in inputs:
        value, k = maj[c]
        note = ""
        if k / len(good) < MAJORITY:
            note = f"  [mixed: only {k} of {len(good)} good builds agree]"
        lines.append(f"  {c} = {value}  ({k} of {len(good)} good builds){note}")

    cands = []
    screened = 0
    for c in inputs:
        vals = [_text(r.get(c)) for r in rows]
        for v in set(vals):
            screened += 1
            with_i = [i for i in usable if vals[i] == v]
            without_i = [i for i in usable if vals[i] != v]
            if len(with_i) < MIN_SUPPORT or len(without_i) < MIN_SUPPORT:
                continue
            mean_with = sum(score[i] for i in with_i) / len(with_i)
            mean_without = sum(score[i] for i in without_i) / len(without_i)
            if mean_with > mean_without:
                cands.append((mean_with - mean_without, c, v, with_i, without_i,
                              mean_with, mean_without))
    cands.sort(key=lambda t: -t[0])

    if not cands:
        lines.append("No single setting is clearly associated with higher goal scores "
                     "in this data.")
    else:
        lines.append(
            f"Settings most associated with a higher goal score (of {screened} setting "
            "values screened; mean score with the setting vs without it):"
        )
        for _, c, v, wi, wo, mw, mo in cands[:TOP_SETTINGS]:
            noise = same([score[i] for i in wi], [score[i] for i in wo])
            verdict = ("not distinguishable from noise by a permutation test" if noise
                       else "unlikely to be noise (permutation test, p below 0.05)")
            lines.append(f"  {c} = {v}: {mw:.2f} with vs {mo:.2f} without "
                         f"(n = {len(wi)} vs {len(wo)}); {verdict}.")

    lines.append(
        "Caveats: these are associations in the training data, not proof that a setting "
        "causes the better score. Each setting is judged on its own, so the combination "
        "recommended above has not been tested as a whole, and with many settings screened "
        "some results can be chance. Confirm the set with fresh measurements before relying on it."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this row (goals included) is good or bad."""
    rows, inputs, goals, good_set = _data(train_csv)
    if not goals:
        return "This file has no goal columns, so the build cannot be rated good or bad."

    o_row = _oriented(row, goals)
    if o_row is None:
        return ("The build cannot be rated: one or more of its goal values is missing "
                "or not a number.")

    oriented = [_oriented(r, goals) for r in rows]
    usable = [i for i, o in enumerate(oriented) if o is not None]
    if not usable:
        return "The training data has no complete goal values, so the build cannot be compared."

    lo, hi = _scale([oriented[i] for i in usable], len(goals))
    s = _score(o_row, lo, hi)
    pct = 100.0 * sum(1 for i in usable if _score(oriented[i], lo, hi) < s) / len(usable)
    dominators = sum(1 for i in usable if _dominates(oriented[i], o_row))

    tail = ""
    if good_set:
        gl = sorted(good_set)
        maj = _majority(rows, inputs, gl)
        differs = [c for c in inputs if _text(row.get(c)) != maj[c][0]]
        if not differs:
            tail = "; its settings match the most common good-build setting on every input"
        else:
            shown = ", ".join(
                f"{c}={_text(row.get(c))} (good builds mostly use {maj[c][0]})"
                for c in differs[:3]
            )
            more = f" and {len(differs) - 3} more" if len(differs) > 3 else ""
            tail = f"; its settings differ from the most common good-build setting in {shown}{more}"

    if dominators == 0:
        return (f"Rated good: no training build is at least as good on every goal and better "
                f"on one, and its goal score {s:.2f} (0 = worst, 1 = best) is higher than "
                f"{pct:.0f}% of training builds{tail}.")
    return (f"Rated bad: {dominators} of {len(usable)} training builds are at least as good on "
            f"every goal and better on one, and its goal score {s:.2f} (0 = worst, 1 = best) "
            f"is higher than only {pct:.0f}% of training builds{tail}.")

