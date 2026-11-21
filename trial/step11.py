"""Step 11: remember. Player types and the strong-and-good players.

Standard library only. Two client questions:
  - families(): the player types, via step1's flag-family clustering (k-means
    on the input columns).
  - best(): the unusually good players, the Pareto front on all goals ranked by
    mean normalised goal score and capped at 5% of rows.

families() is step1's clustering unchanged. best() reuses step1's loading,
column splitting, goal normalisation and dominance test, but builds the Pareto
front incrementally. step1's layered sort ranks every row into all layers,
which is far too slow for 17,000 players, and only the first layer is needed.
"""

import step1
from step1 import _load, _columns, _normalised_goals, _dominates


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return step1.families(train_csv)


def _front(points):
    """Indices of the points that no other point dominates (the Pareto front),
    in ascending index order. Same rule as step1: identical points are all kept.

    Only front members need checking: a point dominated by any earlier point is
    also dominated by a front member, because dominance is transitive."""
    front = []
    for i, p in enumerate(points):
        if any(_dominates(points[j], p) for j in front):
            continue
        front = [j for j in front if not _dominates(p, points[j])]
        front.append(i)
    return sorted(front)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    ranked by mean normalised goal value and capped at 5% of rows."""
    header, rows = _load(train_csv)
    n = len(rows)
    cap = n // 20  # floor(5% of n), computed without float error
    _, goals = _columns(header)
    if n == 0 or cap == 0 or not goals:
        return []

    points = _normalised_goals(rows, goals)
    score = [sum(p) / len(p) for p in points]

    front = _front(points)
    front.sort(key=lambda i: (-score[i], i))
    return front[:cap]

