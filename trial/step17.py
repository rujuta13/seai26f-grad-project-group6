"""step17.py: choose, for the two client questions.

Q1: "Find the best strong penalty-taker while sending scouts to see only 20 players."
    Both Strength+ and Penalties+ are goals, so no single goal is named and the
    search optimises them together (goal=None), by Pareto fronts as in step1.best.
    Call with budget=20.

Q2: "Find the best one under 24, in a league we already scout."
    Age and league are input conditions, not goals, and choose() takes no filter
    argument. The caller must restrict blind_csv to eligible rows (Age < 24 and
    a league already scouted) before calling. The returned index then refers to
    that filtered file.

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
    return _choose(blind_csv, label, budget=budget, goal=goal)

