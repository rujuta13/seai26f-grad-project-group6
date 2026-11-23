"""step27.py: choose which customer to target with a limited test mailing.

Two client questions, both answered by the same budgeted search:

1. "Find the best customers to target while test-mailing only 50 people."
   Call with budget=50 and goal=None, so every goal counts and the answer is
   on the joint Pareto front of the measured customers.

2. "If I care most about store visits rather than web orders, who should I
   target instead?"
   Call with goal="NUMSTOREPURCHASES+" so only in-store purchases are optimised.

The search itself (phased exploration, then refinement near the best customer
found so far, with a hard cap on label() calls) lives in step17.choose, which
wraps step7.choose. This module only supplies the entry point with the
signature the client needs.
"""
from __future__ import annotations

from step17 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, after calling label(i) at most `budget` times.

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column name: value}.
    budget: maximum number of label() calls; a positive integer. Calls beyond
        the budget raise RuntimeError.
    goal: optional name of one goal column; if given, only that goal is optimised.
        Otherwise all goal columns are optimised jointly.
    """
    return int(_choose(blind_csv, label, budget=budget, goal=goal))

