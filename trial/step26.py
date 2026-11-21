"""step26.py: justify, for the two client questions.

Q1: "Explain to the sales director why this group should get the next campaign."
Q2: "Why did you mark this customer as low value? Give me reasons I can act on."

explain() reuses step6.explain, which gives plain-language reasons that make a
row good and cites columns and values that can be checked in the training file.
why() reuses step6.why for the verdict and the weakest goal, and adds the input
value most tied to low-value customers, so the reply names something that can
be acted on. Only the customer wording and that input lookup are new here.

Reading, column splitting, filling, goal vectors, scores, bins, group labels,
the shrunken group score and the wording of the step6 explanations come from
steps 1, 2, 4, 5 and 6.
"""
from __future__ import annotations

import math

from step1 import _columns, _num, _read
from step2 import _fill, _quantile
from step4 import _objectives
from step5 import BAD_QUANTILE, SHRINK, _labels, _scores
from step6 import MIN_GROUP, _describe
from step6 import explain as _explain
from step6 import why as _why


def _tidy(text: str) -> str:
    """Rewrite the step6 wording (builds, benchmarks) as customer wording."""
    swaps = [
        ("Confirm a claim by re-benchmarking before acting on it.",
         "Confirm a claim against fresh customer records before acting on it."),
        ("one benchmark sample", "one sample of customers"),
        ("bad builds", "low-value customers"),
        ("builds", "customers"),
        ("build", "customer"),
    ]
    for old, new in swaps:
        text = text.replace(old, new)
    return text


def _low_input(rows: list[dict], inputs: list[str], bad: list[bool],
               p0: float, f: dict):
    """The input value of customer f that is most tied to low value.

    Groups are the same as step5 (values, or quantile bins for numeric columns
    with many levels). A group is scored by its shrunken excess bad rate over
    the overall rate, as in step5. Returns (excess, column, group label,
    bad count, group size, bad rate) for the best group, or None if no group
    of f's values is more bad than average.
    """
    n = len(rows)
    best_found = None
    for c in inputs:
        labels = _labels(rows + [f], c)
        qlab = labels[n]
        size = 0
        hits = 0
        for lab, is_bad in zip(labels[:n], bad):
            if lab == qlab:
                size += 1
                if is_bad:
                    hits += 1
        if size < MIN_GROUP:
            continue
        rate = hits / size
        excess = (rate - p0) * size / (size + SHRINK)
        if excess > 0 and (best_found is None or excess > best_found[0]):
            best_found = (excess, c, qlab, hits, size, rate)
    return best_found


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes
    customers good. Every claim cites a column and value that can be looked up
    in train_csv, so a group can be judged against the file before a campaign."""
    return _tidy(_explain(train_csv))


def why(train_csv, row) -> str:
    """One or two sentences on why this customer (goals included) is rated good,
    middling or bad. The second sentence names the weakest goal and the input
    value of this customer that is most tied to low-value customers."""
    text = _why(train_csv, row)
    if text.startswith("This build cannot"):
        return _tidy(text)

    # The verdict sentence is everything before step6's Pareto clause.
    verdict = text.split(" Pareto-wise, ", 1)[0]

    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    m = len(goals)

    obj = [_objectives(r, gmax, gmin) for r in rows]
    score = _scores(obj, m)
    cut = _quantile(score, BAD_QUANTILE)
    bad = [s <= cut for s in score]
    p0 = sum(bad) / n

    # Weakest goal of this customer, normalised as in step6.why.
    lo, span = [], []
    for j in range(m):
        fin = [o[j] for o in obj if math.isfinite(o[j])]
        a = min(fin) if fin else 0.0
        b = max(fin) if fin else 0.0
        lo.append(a)
        span.append(b - a)
    vec = _objectives(row, gmax, gmin)
    nv = []
    for j, x in enumerate(vec):
        if not math.isfinite(x):
            nv.append(0.0)
        elif span[j] > 0:
            nv.append(min(1.0, max(0.0, (x - lo[j]) / span[j])))
        else:
            nv.append(1.0)
    j = min(range(m), key=lambda k: nv[k])
    raw = _num(row.get(goals[j]))
    raw_text = f"{raw:g}" if raw is not None else "missing"

    parts = [f"Its weakest goal is {goals[j]} ({raw_text})"]
    f = _fill(row, rows, inputs)
    found = _low_input(rows, inputs, bad, p0, f)
    if found is not None:
        _, c, qlab, hits, size, rate = found
        parts.append(
            f"the input most tied to low-value customers is {_describe(c, qlab)}, "
            f"where {hits} of {size} customers are low value (bottom quarter on "
            f"the goals, {rate:.0%}) against {p0:.0%} overall"
        )
    second = "; ".join(parts) + "."

    return _tidy(verdict + " " + second)

