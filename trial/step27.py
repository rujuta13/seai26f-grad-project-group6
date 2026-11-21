"""step27.py: choose the best customers to test-mail within a fixed mailing budget.

Standard library only (Python 3.13). Reuses step17.choose, which already runs
the two-phase budgeted search from step7 with the scalable clustering from
step11.families.

Client questions:
  - "Find the best customers to target while test-mailing only 50 people":
    choose(blind, label, budget=50).
  - "If I care most about store visits rather than web orders, who should I
    target instead?": choose(blind, label, budget=50,
    goal="NUMSTOREPURCHASES+").
"""
from __future__ import annotations

from step17 import choose as _step17_choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best customer found within `budget` calls to label(i).

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column name: value}.
    budget: at most this many label calls; more raises RuntimeError.
    goal: optimise only this goal (name with or without +/- suffix).
          None optimises all goals together.
    """
    return _step17_choose(blind_csv, label, budget=budget, goal=goal)

