"""step17.py - client question 17: choose.

choose(blind_csv, label, budget=30, goal=None) returns the index of the best
row found while calling label(i) at most `budget` times. blind_csv has every
goal cell set to '?'.

The search is step7.choose: a coverage phase over k-means clusters, then
alternating exploitation and exploration. Choosing among benchmarked rows uses
non-dominance on the chosen goals, then the mean min-max goal score.

Client usage (the caller passes the arguments; this module does not filter
rows by age or by league):
  - best strong penalty-taker, 20 scouting visits:
        choose(blind_csv, label, budget=20, goal="Penalties")
  - best player under 24 in a league already scouted:
        restrict blind_csv to those rows before calling choose, then call
        choose(blind_csv_restricted, label, budget=...)

Reused from earlier steps:
  step7: choose (the full search, budget checks and goal selection)
"""
from step7 import choose as _choose

__all__ = ["choose"]


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found while calling label(i) at most `budget`
    times. label(i) returns the goals of row i as {goal column name: value}.
    If `goal` names one goal, only that goal is optimised; otherwise all goals
    count. Raises an error if label would be called more than `budget` times."""
    return int(_choose(blind_csv, label, budget=budget, goal=goal))

