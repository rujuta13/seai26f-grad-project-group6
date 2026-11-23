"""step27.py: choose the best customers to target while test-mailing a limited number.

Thin wrapper around step17.choose, which checks the arguments and counts calls to
`label` so that a call beyond `budget` raises an error. The search itself is
step7.choose.

The client's two questions map onto this function through its arguments:
- Best customers with 50 test mailings: budget=50 and no goal, so all goals count.
- Store visits instead of web orders: goal="NUMSTOREPURCHASES+" (the store-visit goal).
"""

from step17 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, calling label(i) at most `budget` times.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict. If `goal` names one goal, only that goal is optimised.
    Raises RuntimeError if the search would call label more than `budget` times.
    """
    return int(_choose(blind_csv, label, budget=budget, goal=goal))

