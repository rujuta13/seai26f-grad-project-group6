"""Step 27: choose. Find the best customers to target within a test-mailing budget.

Standard library only. This is step 17's choose() with the same signature,
which already enforces the budget with a call counter. The client's two
questions map onto its arguments:
  - "best customers, test-mailing only 50 people": budget=50, goal=None.
  - "care most about store visits rather than web orders": goal="NUMSTOREPURCHASES+",
    so only in-store purchases are optimised.
"""

from step17 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Row index of the best customer found, using at most `budget` label calls.

    label(i) returns the goals of row i as a dict. If `goal` names one goal,
    only that goal is optimised. Raises RuntimeError if label would be called
    more than `budget` times.
    """
    return _choose(blind_csv, label, budget=budget, goal=goal)

