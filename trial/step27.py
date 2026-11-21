"""Step 27: find the best customers to target while test-mailing only a few people.

choose() is the client-facing entry point for step7's budgeted search. It
delegates to step7.choose, which holds the implementation. The search sees the
flag settings of every row of blind_csv but none of the goal values, calls
label(i) at most `budget` times, and returns the index of the best row found.

Client use:
  * "Best customers to target, test-mailing only 50 people":
        choose(blind_csv, label, budget=50)
  * "Who should I target if store visits matter more than web orders?":
        choose(blind_csv, label, budget=50, goal="NUMSTOREPURCHASES+")

Raises ValueError for an unknown goal, and RuntimeError if label would be
called more than `budget` times.
"""

from step7 import choose as _choose_step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, calling label() at most budget times.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of row i
    as a dict. If goal names one goal, only that goal is optimised.
    """
    return int(_choose_step7(blind_csv, label, budget=budget, goal=goal))

