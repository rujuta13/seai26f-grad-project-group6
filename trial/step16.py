"""step16.py: justify the recommended player and rate single players.

explain(train_csv): reasons, in plain words a skeptical expert can check on
video, for what makes players good. The recommended player is the best player
on the Pareto front (no training player is at least as good on every goal and
better on one), ranked by normalised goal sum as in step1.best. Attribute
settings are compared by their share among front players and among all players.
A setting is reported as shifting a goal only when step3.same finds that goal
difference real.

why(train_csv, row): one or two sentences rating a single player. A player is
good when no training player beats them (the step6.why test). Each goal is then
compared with the calibrated 10th-90th percentile range from step12.guess, so a
wide range means low confidence.

For about 17,000 players this uses step11's fast front (step1.best is O(n^2)),
skips rows with non-finite goals, and caps the samples given to step3.same,
whose Cliff's delta costs O(n*m). Standard library only.
"""
import math
import random
import statistics
from collections import Counter

from step1 import _load, _columns, _dominates, BEST_FRACTION
from step3 import same
from step5 import CATEGORY_LIMIT
from step6 import MIN_SHARE_GAP, MAX_REASONS, _normalised_score, _predicted
from step11 import _pareto_front, _top_by_score
from step12 import guess
from step14 import _orient

SAMPLE_CAP = 400   # largest group passed to step3.same (keeps its O(n*m) cost small)
SAMPLE_SEED = 1    # fixed seed so results are reproducible


def _fmt(v):
    """Format a number compactly; other cells are shown as they are."""
    return f"{v:g}" if isinstance(v, float) else str(v)


def _sample(values, seed):
    """At most SAMPLE_CAP values, drawn with a fixed seed."""
    if len(values) <= SAMPLE_CAP:
        return values
    return random.Random(seed).sample(values, SAMPLE_CAP)


def explain(train_csv) -> str:
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    n = len(rows)
    if n == 0 or not goals:
        return ("There is no recommended player: the training file has no rows "
                "or no goal columns.")

    V = [_orient(r, goals) for r in rows]
    usable = [i for i in range(n) if V[i] is not None]
    if not usable:
        return ("There is no recommended player: no training player has usable "
                "values for every goal.")

    # Pareto front and its cap, as in step1.best, computed on the usable rows.
    cap = int(BEST_FRACTION * n)
    UV = [V[i] for i in usable]
    front_pos = _pareto_front(UV)
    if len(front_pos) > cap:
        front_pos = _top_by_score(UV, front_pos, cap)
    front = sorted(usable[p] for p in front_pos)
    if not front:
        return ("There is no recommended player: too few usable training players "
                "to form a Pareto front.")

    score = _normalised_score(UV)
    rec = max(front, key=lambda i: (score(V[i]), -i))
    nf = len(front)
    nu = len(usable)

    lines = [
        f"Recommended player: training row {rec} (counting from 0). It is one of "
        f"{nf} players on the Pareto front: no training player is at least as good "
        "on every goal and better on at least one.",
        "Attributes: " + ", ".join(f"{c}={_fmt(rows[rec][c])}" for c in inputs) + ".",
        "Goals, recommended player against all players (medians):",
    ]
    for c, sgn in goals:
        raw_all = [float(rows[i][c]) for i in usable]
        raw_front = [float(rows[i][c]) for i in front]
        direction = "higher" if sgn > 0 else "lower"
        lines.append(
            f"- {c}: recommended {float(rows[rec][c]):g}; front median "
            f"{statistics.median(raw_front):g}; all-player median "
            f"{statistics.median(raw_all):g}; {direction} is better."
        )

    # For each input with few distinct values, find the setting whose share
    # among front players differs most from its share among all players.
    cands = []
    for c in inputs:
        count_all = Counter(rows[i][c] for i in usable)
        if not count_all or len(count_all) > CATEGORY_LIMIT:
            continue  # continuous input: no shares to compare
        count_front = Counter(rows[i][c] for i in front)
        gap, v, sf, sa = max(
            ((abs(count_front.get(v, 0) / nf - cnt / nu),
              v, count_front.get(v, 0) / nf, cnt / nu)
             for v, cnt in count_all.items()),
            key=lambda t: t[0],
        )
        cands.append((gap, c, v, sf, sa))
    cands.sort(key=lambda t: -t[0])

    lines.append("Attribute settings that set the front apart "
                 "(share among front players vs among all players):")
    shown = 0
    for gap, c, v, sf, sa in cands:
        if gap < MIN_SHARE_GAP or shown == MAX_REASONS:
            break
        shown += 1
        inside = [i for i in usable if rows[i][c] == v]
        outside = [i for i in usable if rows[i][c] != v]
        checks = []
        for g, _ in goals:
            xs = _sample([float(rows[i][g]) for i in inside], SAMPLE_SEED)
            ys = _sample([float(rows[i][g]) for i in outside], SAMPLE_SEED)
            real = bool(xs and ys) and not same(xs, ys)
            checks.append(f"{g} {'differs' if real else 'shows no real difference'}")
        check_text = "; ".join(checks)
        lines.append(
            f"- {c}={_fmt(v)}: {sf:.0%} of front players vs {sa:.0%} of all players. "
            f"Players with it vs without it: {check_text}."
        )
    if shown == 0:
        lines.append(f"- No single setting differs by {MIN_SHARE_GAP:.0%} or more "
                     "between front players and all players.")

    lines.append(
        "These are associations in the training data: they show where good players "
        "sit, not that an attribute causes the difference. Check the reasons on "
        "video and against more matches before relying on them."
    )
    return "\n".join(lines)


def why(train_csv, row) -> str:
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals:
        return "This training file has no goal columns, so the player cannot be rated."

    pred = guess(train_csv, row)
    V = _orient(row, goals)
    if V is None:
        missing = [c for c, _ in goals
                   if not isinstance(row.get(c), float) or not math.isfinite(row[c])]
        predicted = "; ".join(_predicted(c, pred[c]) for c, _ in goals)
        return ("This player cannot be rated: no usable value for "
                f"{', '.join(missing)}. The nearest training players predict "
                f"{predicted}.")

    train_vecs = [v for v in (_orient(r, goals) for r in rows) if v is not None]
    if not train_vecs:
        return ("This player cannot be rated: the training file has no usable "
                "players to compare with.")
    n = len(train_vecs)
    dominators = sum(1 for t in train_vecs if _dominates(t, V))
    if dominators == 0:
        verdict = (f"This player is good: none of the {n} training players is at "
                   "least as good on every goal and better on at least one, so this "
                   "player is on the Pareto front")
    else:
        verdict = (f"This player is bad: {dominators} of the {n} training players are "
                   "at least as good on every goal and better on at least one, so "
                   "this player is beaten")

    parts = []
    for c, sgn in goals:
        est, lo, hi = pred[c]
        value = float(row[c])
        o = sgn * value
        olo, ohi = sorted((sgn * lo, sgn * hi))
        if o > ohi:
            rel = "is better than"
        elif o < olo:
            rel = "is worse than"
        else:
            rel = "is within"
        direction = "lower" if sgn < 0 else "higher"
        parts.append(
            f"{c} = {value:g} {rel} the calibrated range {lo:g} to {hi:g} "
            f"(median {est:g}; {direction} is better)"
        )
    return (f"{verdict}. Against the nearest training players, "
            + "; ".join(parts) + ".")

