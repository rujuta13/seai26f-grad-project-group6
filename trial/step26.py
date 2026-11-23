"""step26.py: customer-facing justification for the campaign and low-value questions.

explain() says why the recommended group of customers should get the next campaign.
why() gives one verdict sentence and actionable reasons for a single customer.

Campaign-acceptance columns are left out of the trait reasons. They describe the same
behaviour as the goals, so they would trivially rank first and tell the client nothing
new. Everything else is reused from earlier steps: loading, classification and goal
matrices from step1; the permutation test from step3; the keep/discard decision from
step4.stream; the good-group, trait-profile and threshold helpers from step16; the
identifier filter from step11; and the campaign column list from step25.
"""
from __future__ import annotations

import numpy as np

from step1 import _classify, _goal_matrix, _load, _to_value
from step3 import same
from step4 import stream
from step11 import _trait_inputs
from step16 import _MIN_GAP_SD, _TOP_TRAITS, _WHY_TRAITS, _good_rows, _profile
from step25 import _CAMPAIGN_COLS

_DEFINITION = (
    "Definition: a customer is in the recommended group if no other customer is at least "
    "as good on every goal and strictly better on at least one (the Pareto front). If the "
    "front is larger than 5% of customers, the ones with the best mean rank across goals are kept."
)

_CAVEAT = (
    "Caveats: these are associations in one customer file, not proof that a trait causes "
    "a customer to buy or respond. Campaign-acceptance history is left out of the trait "
    "reasons because it describes the same behaviour as the goals. Test the campaign on a "
    "held-out group of customers before scaling it up."
)


def _traits(header, inputs):
    """Trait inputs (identifiers removed) without the campaign-acceptance history."""
    return [c for c in _trait_inputs(header, inputs) if header[c] not in _CAMPAIGN_COLS]


def _goal_text(header, cols, row):
    return ", ".join(f"{header[c]}={float(row[header[c]]):g}" for c in cols)


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for why the recommended group is good."""
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")
    maximise, minimise, inputs = _classify(header)
    goal_cols = minimise + maximise
    if not goal_cols:
        return "The file has no goal columns, so no customer group can be called good."

    n = len(rows)
    G = _goal_matrix(rows, maximise, minimise)
    good_idx = _good_rows(train_csv, rows, G)
    good_mask = np.zeros(n, dtype=bool)
    good_mask[good_idx] = True

    lines = [
        _DEFINITION,
        f"{int(good_mask.sum())} of {n} customers form the recommended group.",
    ]

    if good_mask.all():
        lines.append(
            "Every customer is in the recommended group, so it cannot be compared with other customers."
        )
        lines.append(_CAVEAT)
        return "\n".join(lines)

    lines.append("Behaviour of the recommended group against other customers:")
    for col in goal_cols:
        name = header[col]
        direction = "lower is better" if col in minimise else "higher is better"
        vals = np.array([float(r[col]) for r in rows], dtype=float)
        g = vals[good_mask]
        r = vals[~good_mask]
        verdict = "a significant gap" if not same(g, r) else "no significant gap"
        lines.append(
            f"  {name} ({direction}): recommended group median {float(np.median(g)):g}, "
            f"other customers median {float(np.median(r)):g} ({verdict}, permutation test)"
        )

    profile = _profile(header, rows, _traits(header, inputs), good_mask)
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
        lines.append("Customer traits that set the recommended group apart (checkable in the CRM):")
        for p in chosen:
            direction = "higher" if p["gap"] > 0 else "lower"
            lines.append(
                f"  {p['name']}: the recommended group is {direction}, median {p['good']:g} "
                f"against {p['rest']:g} for other customers"
            )
    else:
        lines.append(
            "No customer trait separates the recommended group from the rest by a significant, sizeable gap."
        )

    lines.append(_CAVEAT)
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this customer (goals included) is rated high or low value.

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
    goals = _goal_text(header, goal_cols, row)

    good_idx = _good_rows(train_csv, rows, G)
    good_mask = np.zeros(n, dtype=bool)
    good_mask[good_idx] = True
    profile = _profile(header, rows, _traits(header, inputs), good_mask)

    # This customer's trait values against the recommended group's medians.
    traits_vs_good = []
    for p in profile:
        raw = row.get(p["name"])
        if raw is None:
            continue
        rv = _to_value(raw)
        if isinstance(rv, float):
            traits_vs_good.append((p, (rv - p["good"]) / p["sd"], rv))

    kept = bool(stream(train_csv, [row])[0])

    if kept:
        s1 = (
            f"Rated high value: no training customer is at least as good on every goal and "
            f"strictly better on one ({goals}), so this customer is on the Pareto front."
        )
        strong = sorted(
            (t for t in traits_vs_good if t[1] >= _MIN_GAP_SD),
            key=lambda t: (-t[1], t[0]["col"]),
        )[:_WHY_TRAITS]
        if strong:
            s2 = "Checkable strengths against the recommended group: " + ", ".join(
                f"{p['name']} {rv:g} (group median {p['good']:g})" for p, _, rv in strong
            ) + "."
        else:
            s2 = "No single customer trait stands out above the recommended group's medians."
        return f"{s1} {s2}"

    dominators = np.flatnonzero((G <= gv).all(axis=1) & (G < gv).any(axis=1))
    k = int(dominators[0]) if dominators.size else -1

    # Goals where this customer falls short of the recommended group's median.
    shortfalls = []
    for col in goal_cols:
        vals = np.array([float(r[col]) for r in rows], dtype=float)
        good_med = float(np.median(vals[good_mask]))
        value = float(row[header[col]])
        if (col in minimise and value > good_med) or (col in maximise and value < good_med):
            shortfalls.append(f"{header[col]} {value:g} (group median {good_med:g})")

    s1 = (
        f"Rated low value: training customer {k} is at least as good on every goal and strictly "
        f"better on at least one, so this customer ({goals}) adds nothing to the Pareto front."
    )
    weak = sorted(
        (t for t in traits_vs_good if t[1] <= -_MIN_GAP_SD),
        key=lambda t: (t[1], t[0]["col"]),
    )[:_WHY_TRAITS]
    parts = []
    if shortfalls:
        parts.append("Goals below the recommended group: " + ", ".join(shortfalls[:_WHY_TRAITS]))
    if weak:
        parts.append("Checkable traits below the recommended group: " + ", ".join(
            f"{p['name']} {rv:g} (group median {p['good']:g})" for p, _, rv in weak
        ))
    s2 = (". ".join(parts) + ".") if parts else "No single customer trait falls clearly below the recommended group's medians."
    return f"{s1} {s2}"

