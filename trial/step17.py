"""Step 17: client-facing search for one build under a label budget.

choose(blind_csv, label, budget=30, goal=None): returns the index of the best
row found, calling label(i) at most `budget` times. If `goal` names one goal,
only that goal is optimised; otherwise all goals are optimised together.

Client requests:
  * "Find the best strong penalty-taker while sending scouts to see only 20
    players." -> choose(blind_csv, label, budget=20, goal="Penalties+")
  * "Find the best one under 24, in a league we already scout." -> not covered
    by this interface. It needs two filters, Age under 24 and league in the
    set of leagues already scouted, and that league list is not supplied.
    Restrict blind_csv to matching rows before calling choose, once the league
    list is known. Then call choose(blind_csv, label, budget=...) as usual.
"""

import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, calling label(i) at most `budget` times.

    blind_csv has every goal cell set to '?'. label(i) returns {goal: value} for
    row i. If `goal` names one goal, only that goal is optimised; otherwise all
    goals are optimised together. Calling label more than `budget` times raises
    RuntimeError.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")

    calls = 0

    def counted_label(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("label budget exhausted")
        calls += 1
        return label(i)

    return int(step7.choose(blind_csv, counted_label, budget=budget, goal=goal))

