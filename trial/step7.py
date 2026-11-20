"""Step 7: find a good flag set while benchmarking at most `budget` builds.

The blind file has the input flags of every build but no goal values. Goal
values are only available through label(i), which may be called at most
`budget` times.

choose() works in two phases:

1. Spread: step1.families() splits the builds into clusters. The first share
   of the budget is spent round-robin across clusters, each time on the build
   farthest from every build already labelled (maximin), so every region of
   the flag space is sampled.
2. Exploit and explore: each unlabelled build is scored by the mean goal score
   of its nearest labelled builds plus a bonus for being far from all labelled
   builds (an upper-confidence rule). The highest-scoring candidate is labelled
   next.

Goal scores are normalised over the labelled builds with higher always better,
so several goals are combined as in step1.best. If `goal` names one goal, only
that goal is optimised. The returned index is the labelled build with the best
score; ties go to the lower index.
"""
import math

from step1 import _load, families

_INIT_SHARE = 1.0 / 3.0   # share of the budget spent on spreading out
_K_NEAR = 5               # labelled neighbours used to predict a build's score
_KAPPA = 0.5              # weight of the distance bonus (exploration)


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _goal_specs(header, goal):
    """Return [(column, sign)] for the goals to optimise (+1 maximise, -1 minimise)."""
    specs = []
    for c in header:
        if c.endswith("+"):
            specs.append((c, 1.0))
        elif c.endswith("-"):
            specs.append((c, -1.0))
    if not specs:
        raise ValueError("blind_csv has no goal columns")
    if goal is None:
        return specs
    for c, sg in specs:
        if c == goal or c[:-1] == goal:
            return [(c, sg)]
    raise ValueError(
        f"unknown goal {goal!r}; goals are {[c for c, _ in specs]}"
    )


def _col_spans(rows, cols):
    """Range of each numeric input column over the blind rows (0 if none)."""
    spans = {}
    for c in cols:
        vals = [
            r[c] for r in rows
            if isinstance(r.get(c), float) and math.isfinite(r[c])
        ]
        spans[c] = (max(vals) - min(vals)) if vals else 0.0
    return spans


def _dist(a, b, cols, spans):
    """Mean per-column difference in [0, 1]: scaled gap for numbers, 0/1 otherwise."""
    if not cols:
        return 0.0
    d = 0.0
    for c in cols:
        x = a.get(c)
        y = b.get(c)
        if isinstance(x, float) and isinstance(y, float):
            s = spans[c]
            if s > 0:
                d += min(1.0, abs(x - y) / s)
        elif x != y:
            d += 1.0
    return d / len(cols)


def _score_labeled(order, labeled, specs):
    """Return {row index: score in 0..1} for labelled rows with all goals finite.

    Each goal is normalised over the labelled rows (higher is better) and the
    scores are averaged over the goals being optimised.
    """
    vecs = {}
    for i in order:
        vec = []
        for c, sg in specs:
            f = _finite(labeled[i].get(c))
            if f is None:
                break
            vec.append(sg * f)
        else:
            vecs[i] = vec
    if not vecs:
        return {}

    g = len(specs)
    lo = [min(v[j] for v in vecs.values()) for j in range(g)]
    hi = [max(v[j] for v in vecs.values()) for j in range(g)]
    out = {}
    for i, v in vecs.items():
        total = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            total += 1.0 if span == 0 else (v[j] - lo[j]) / span
        out[i] = total / g
    return out


def _predict(j, valid, lab_dist, mind):
    """Acquisition value of unlabelled row j.

    valid is a list of (position in labelling order, score). The prediction is
    the mean score of the nearest labelled neighbours; the bonus is the distance
    to the nearest labelled row.
    """
    near = sorted(valid, key=lambda t: (lab_dist[j][t[0]], t[0]))[:_K_NEAR]
    pred = sum(s for _, s in near) / len(near)
    return pred + _KAPPA * mind[j]


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found after labelling at most `budget` rows.

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns {goal column: value} for row i; called at most `budget` times.
    goal: optional goal column (or its name without the +/- suffix) to optimise
          alone; if None, all goals are combined.

    Raises ValueError for budget < 1, an empty file, or an unknown goal.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")
    header, rows = _load(blind_csv)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no rows")
    specs = _goal_specs(header, goal)

    cols = [c for c in header if _is_input(c)]
    spans = _col_spans(rows, cols)
    fam = families(blind_csv)

    unlab = set(range(n))
    remaining = {}
    for i, f in enumerate(fam):
        remaining.setdefault(f, []).append(i)
    fam_keys = list(remaining)

    order = []                       # labelled indices, in labelling order
    labeled = {}                     # index -> goal dict from label()
    lab_dist = [[] for _ in range(n)]  # lab_dist[k][p]: distance from k to order[p]
    mind = [math.inf] * n            # distance from each row to nearest labelled row

    def _take(i):
        labeled[i] = label(i) or {}
        order.append(i)
        unlab.discard(i)
        remaining[fam[i]].remove(i)
        for k in range(n):
            d = _dist(rows[k], rows[i], cols, spans)
            lab_dist[k].append(d)
            if d < mind[k]:
                mind[k] = d

    # Phase 1: spread the first share of the budget across clusters.
    init_n = min(budget, max(1, round(budget * _INIT_SHARE)))
    turn = 0
    while len(order) < init_n:
        live = [f for f in fam_keys if remaining[f]]
        if not live:
            break
        f = live[turn % len(live)]
        turn += 1
        k = max(remaining[f], key=lambda j: (mind[j], -j))
        _take(k)

    # Phase 2: label the most promising remaining build, one at a time.
    while len(order) < budget and unlab:
        scores = _score_labeled(order, labeled, specs)
        if scores:
            pos = {i: p for p, i in enumerate(order)}
            valid = [(pos[i], s) for i, s in scores.items()]
            k = max(unlab, key=lambda j: (_predict(j, valid, lab_dist, mind), -j))
        else:
            k = max(unlab, key=lambda j: (mind[j], -j))
        _take(k)

    scores = _score_labeled(order, labeled, specs)
    if not scores:
        return order[0]
    return max(scores, key=lambda i: (scores[i], -i))

