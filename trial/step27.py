"""step27.py: choose, for the two client questions.

Q1: "Find the best customers to target while test-mailing only 50 people."
    No single goal is named, so the search optimises all goals together
    (goal=None), by Pareto fronts as in step1.best. Call with budget=50.

Q2: "If I care most about store visits rather than web orders, who should I
    target instead?"
    Call with goal="NUMSTOREPURCHASES+" (or the bare name "NUMSTOREPURCHASES").
    Only that goal is then optimised.

The search, the label budget and the leader rule come from step7.choose, which
raises RuntimeError if label is called more than budget times.
"""
from __future__ import annotations

from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, calling label at most `budget` times.

    blind_csv: a MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column: value}.
    budget: the most label calls allowed. Going over raises RuntimeError.
    goal: optional goal column name (or its name without the trailing + or -).
          When given, only that goal is optimised. When None, all goals are
          optimised together by Pareto fronts.
    """
    return int(_choose(blind_csv, label, budget=budget, goal=goal))

