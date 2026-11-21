"""step27.py - client question 27: choose.

choose(blind_csv, label, budget=30, goal=None) returns the index of the best
row found while calling label(i) at most `budget` times. blind_csv has every
goal cell set to '?'.

The search is step7.choose: a coverage phase over k-means clusters, then
alternating exploitation and exploration. With goal=None all goals count.
With goal set to one goal, only that goal is optimised. The goal may be
named with or without its '+'/'-' suffix.

Client usage (the caller passes the arguments; this module does not restrict
rows):
  - best customers to target while test-mailing only 50 people:
        choose(blind_csv, label, budget=50)
  - target store visits rather than web orders (NUMSTOREPURCHASES+ is the
    in-store purchase goal, NUMWEBPURCHASES+ the web one):
        choose(blind_csv, label, budget=50, goal="NUMSTOREPURCHASES+")

Reused from earlier steps:
  step7: choose (the full search, budget checks, goal selection)
"""
from step7 import choose as _choose

__all__ = ["choose"]


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found while calling label(i) at most `budget`
    times. label(i) returns the goals of row i as {goal column name: value}.
    If `goal` names one goal, only that goal is optimised; otherwise all goals
    count. Raises an error if label would be called more than `budget` times."""
    return int(_choose(blind_csv, label, budget=budget, goal=goal))

