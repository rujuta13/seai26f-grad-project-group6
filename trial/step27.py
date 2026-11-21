"""Step 27: pick the best customers to target with a limited test mailing.

choose(blind_csv, label, budget=30, goal=None) is the scout-limited search
from step17. It labels at most `budget` customers (for example, a test
mailing to 50 people) and returns the index of the best one found.

If `goal` names one goal, only that goal is optimised. For example,
goal="NUMSTOREPURCHASES" (or "NUMSTOREPURCHASES+") targets store visits
rather than web orders, which would be goal="NUMWEBPURCHASES".

Calls to label() beyond the budget raise RuntimeError, as in step17.
"""
from step17 import choose as _choose17


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found after at most `budget` calls to label().

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns {goal column: value} for row i.
    budget: maximum number of label() calls. Exceeding it raises RuntimeError.
    goal: optional goal column (or its name without the +/- suffix) to optimise
          alone. If None, all goals are combined.

    Raises ValueError for budget < 1, an empty file, or an unknown goal.
    """
    return _choose17(blind_csv, label, budget=budget, goal=goal)

