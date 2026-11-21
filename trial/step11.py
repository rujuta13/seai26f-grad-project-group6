"""step11.py: remember. Sort players into the types they really are, and show
the players who are strong and good from the spot.

families() clusters players on how they play and what they look like. Columns
that only name, link or number a player, or say which club or country they
belong to, are left out, so the clusters reflect play styles rather than
labels. best() is step1.best: the Pareto-ranked rows over all goals together
(here Strength+ and Penalties+, the "strong and good from the spot" players).

Reading, column splitting, feature scaling, k-means and the Pareto best set
come from step1. Only the choice of clustering columns is new here.
"""
from __future__ import annotations

from step1 import K_FAMILIES, _columns, _features, _kmeans, _read
from step1 import best as _pareto_best

# Columns that name, link or number a player, or say which club or country the
# player belongs to. They do not describe how a player plays, so they are not
# used to form families.
IDENTITY_COLUMNS = frozenset({
    "num", "Rank", "Name", "url", "Nation", "League", "Team",
})


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order.

    Families are k-means clusters over the input columns that describe play
    (IDENTITY_COLUMNS are left out). Goal columns are never used.
    """
    header, rows = _read(str(train_csv))
    inputs, _, _ = _columns(header)
    cols = [c for c in inputs if c not in IDENTITY_COLUMNS]
    if not rows:
        return []
    if not cols:
        return [0] * len(rows)
    return _kmeans(_features(rows, cols), K_FAMILIES)


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows, best first. Delegates to step1.best.
    """
    return _pareto_best(train_csv)

