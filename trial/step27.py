"""step27.py: choose customers to target with a fixed mailing budget.

choose(blind_csv, label, budget=30, goal=None) maps the client's two requests
onto the search:
  - "Find the best customers to target while test-mailing only 50 people":
    choose(blind_csv, label, budget=50)
  - "If I care most about store visits rather than web orders, who should I
    target instead?": choose(blind_csv, label, budget=50,
    goal="NUMSTOREPURCHASES+"). The goal name may be given with or without
    its +/- suffix, so "NUMSTOREPURCHASES" also works.

The search is step17.choose, which is step7.choose (explore across customer
segments, then exploit by predicted score). It is reused unchanged. This module
adds only a check that the caller's label function is never asked for more
than `budget` customers. Standard library only.
"""
from step17 import choose as _search


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best customer found, mailing (calling label on) at most
    `budget` customers.

    label(i) returns the goals of row i as a dict and may be called at most
    `budget` times; a further call raises RuntimeError. If `goal` names one
    goal column (with or without its +/- suffix), only that goal is optimised.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a whole number of at least 1")

    calls = 0

    def counted_label(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("benchmark budget exhausted")
        calls += 1
        return label(i)

    return _search(blind_csv, counted_label, budget=budget, goal=goal)

