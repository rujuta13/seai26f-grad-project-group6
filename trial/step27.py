"""Step 27: client-facing search for customers to target under a mailing limit.

choose(blind_csv, label, budget=30, goal=None): returns the index of the best
customer found, calling label(i) at most `budget` times. If `goal` names one
goal, only that goal is optimised; otherwise all goals are optimised together.

Client requests:
  * "Find the best customers to target while test-mailing only 50 people."
      -> choose(blind_csv, label, budget=50)
  * "If I care most about store visits rather than web orders, who should I
     target instead?"
      -> choose(blind_csv, label, budget=50, goal="NUMSTOREPURCHASES+")
"""

import step17


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, calling label(i) at most `budget` times.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of row i
    as a dict {goal: value}. If `goal` names one goal, only that goal is
    optimised; otherwise all goals are optimised together. Calling label more
    than `budget` times raises RuntimeError.
    """
    return int(step17.choose(blind_csv, label, budget=budget, goal=goal))

