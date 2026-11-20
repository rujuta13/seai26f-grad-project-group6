"""step6.py: justify. Plain-language reasons for what makes a build good, and
a one-sentence verdict on any single build.

Reading, column splitting, number parsing, quantiles, goal scores, blame labels,
the Pareto best set and the dominance test come from steps 1, 2, 4 and 5.
Only the explanation text is new here.
"""
from __future__ import annotations

import math

from step1 import BEST_PERCENT, _columns, _num, _read, best as pareto_best
from step2 import _quantile
from step4 import _dominates, _objectives
from step5 import BAD_QUANTILE, BINS, SHRINK, _labels, _scores, blame

TOP_INPUTS = 3      # input columns cited in explain()
MIN_GROUP = 3       # a value must occur at least this often to be cited


def _fmt(v) -> str:
    """Show a cell value as plain text."""
    if isinstance(v, float):
        return f"{v:g}"
    if v is None or v == "":
        return "missing"
    return str(v)


def _describe(c: str, lab) -> str:
    """Plain-language name for a group label produced by step5._labels."""
    if isinstance(lab, tuple):
        return f"{c} in quantile bin {lab[1] + 1} of {BINS} (low to high)"
    if isinstance(lab, float):
        return f"{c} = {lab:g}"
    return f"{c} = {lab}"


def _lifts(rows: list[dict], c: str, good: set[int]) -> list[tuple]:
    """Values of input column c that are more common among good builds than
    overall. Each entry is (score, description, share of the value's builds
    that are good, number of builds with the value). Shrunk as in step5, so
    small groups count for less. Best first."""
    n = len(rows)
    p0 = len(good) / n
    labels = _labels(rows, c)
    size: dict = {}
    hits: dict = {}
    for i, lab in enumerate(labels):
        size[lab] = size.get(lab, 0) + 1
        if i in good:
            hits[lab] = hits.get(lab, 0) + 1
    out = []
    for lab, t in size.items():
        if t < MIN_GROUP:
            continue
        rate = hits.get(lab, 0) / t
        score = (rate - p0) * t / (t + SHRINK)
        if score > 0:
            out.append((score, _describe(c, lab), rate, t))
    out.sort(key=lambda x: -x[0])
    return out


def explain(train_csv) -> str:
    """Plain-language reasons for what makes builds good. Every claim can be
    checked against the training file."""
    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    if n == 0 or not goals:
        return ("No explanation is possible: the data has no goal columns or no "
                "rows, so no build can be called good.")

    ranked = pareto_best(train_csv)
    if not ranked:
        return "No explanation is possible: no build could be ranked on the goals."
    good = set(ranked)
    p0 = len(good) / n
    rec = rows[ranked[0]]

    lines = [
        "1. What 'good' means. A build is good if no other build is at least as "
        "good on every goal and strictly better on at least one (Pareto dominance). "
        f"The {len(good)} best builds out of {n} ({p0:.0%}) are taken from the top "
        f"Pareto fronts, capped at {BEST_PERCENT}% of rows, and ordered within each "
        "front by mean normalised goal score.",
        "",
        "2. Goal values of the recommended build (first in the ranking) against "
        "the median of all builds:",
    ]
    for g in goals:
        direction = "higher is better" if g in gmax else "lower is better"
        v = _num(rec.get(g))
        vals = [x for x in (_num(r.get(g)) for r in rows) if x is not None]
        if v is None or not vals:
            lines.append(f"   {g}: no usable value.")
        else:
            lines.append(
                f"   {g} = {v:g} (median {_quantile(vals, 0.5):g}; {direction})."
            )

    settings = ", ".join(f"{c}={_fmt(rec.get(c))}" for c in inputs)
    lines += [
        "",
        "3. Flag settings of the recommended build:",
        "   " + (settings if settings else "(no input columns)"),
        "",
        "4. Input values that stand out among good builds:",
    ]

    stand = []
    for idx, c in enumerate(inputs):
        found = _lifts(rows, c, good)
        if found:
            stand.append((found[0], idx))
    stand.sort(key=lambda t: (-t[0][0], t[1]))
    if not stand:
        lines.append("   None: no input value is clearly more common among good builds.")
    for (_, desc, rate, size), _ in stand[:TOP_INPUTS]:
        lines.append(
            f"   {desc}: {size} builds have it, {rate:.0%} of them are good, "
            f"against {p0:.0%} overall."
        )

    names = blame(train_csv)[:TOP_INPUTS]
    lines += [
        "",
        "5. Cross-check from the blame ranking (columns most tied to bad builds): "
        + (", ".join(names) if names else "none") + ".",
        "",
        "These are associations in one benchmark sample, not proven causes. "
        "Confirm a claim by re-benchmarking before acting on it.",
    ]
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences: whether this build is good or bad, and why."""
    header, rows = _read(str(train_csv))
    _, gmax, gmin = _columns(header)
    goals = gmax + gmin
    if not rows or not goals:
        return "This build cannot be rated: the data has no goal columns or no rows."

    vec = _objectives(row, gmax, gmin)
    if not any(math.isfinite(x) for x in vec):
        return "This build cannot be rated: it has no usable goal values."

    m = len(goals)
    obj = [_objectives(r, gmax, gmin) for r in rows]

    # Normalise this build's goals with the same bounds used for training scores.
    lo, span = [], []
    for j in range(m):
        fin = [o[j] for o in obj if math.isfinite(o[j])]
        a = min(fin) if fin else 0.0
        b = max(fin) if fin else 0.0
        lo.append(a)
        span.append(b - a)

    nv = []
    for j, x in enumerate(vec):
        if not math.isfinite(x):
            nv.append(0.0)
        elif span[j] > 0:
            nv.append(min(1.0, max(0.0, (x - lo[j]) / span[j])))
        else:
            nv.append(1.0)
    s = sum(nv) / m

    scores = _scores(obj, m)
    lower = _quantile(scores, BAD_QUANTILE)
    upper = _quantile(scores, 1.0 - BAD_QUANTILE)
    if s <= lower:
        verdict = "bad"
    elif s >= upper:
        verdict = "good"
    else:
        verdict = "middling"

    first = (
        f"This build is rated {verdict}: its mean normalised goal score is "
        f"{s:.2f}, against a bottom-quarter cutoff of {lower:.2f} and a "
        f"top-quarter cutoff of {upper:.2f} over the {len(rows)} training builds."
    )

    dominators = sum(1 for o in obj if _dominates(o, vec))
    if dominators == 0:
        pareto = ("no training build is at least as good on every goal and "
                  "better on one")
    else:
        pareto = (f"{dominators} of {len(rows)} training builds are at least as "
                  "good on every goal and better on one")
    j = min(range(m), key=lambda k: nv[k])
    raw = _num(row.get(goals[j]))
    raw_text = f"{raw:g}" if raw is not None else "missing"
    second = f"Pareto-wise, {pareto}; its weakest goal is {goals[j]} ({raw_text})."
    return first + " " + second

