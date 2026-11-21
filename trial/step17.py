"""Step 17: choose. Find the best build within a scouting budget.

Standard library only. This is step 7's choose() with the same signature. The
client's two questions map onto its arguments:
  - "best strong penalty-taker, seeing only 20 players": goal="Penalties+",
    budget=20.
  - "best one under 24, in a league we already scout": the caller passes a
    blind_csv that already holds only the under-24 players of that league, then
    calls choose() as usual.

Every label call goes through a counter, so the budget is enforced here as
well as in step 7. Calling label more than `budget` times raises RuntimeError.
"""

from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Row index of the best build found, using at most `budget` label calls.

    label(i) returns the goals of row i as a dict. If `goal` names one goal,
    only that goal is optimised. Raises RuntimeError if label would be called
    more than `budget` times.
    """
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a whole number of at least 1")

    calls = 0

    def counted(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("benchmark budget exhausted")
        calls += 1
        return label(i)

    return _choose(blind_csv, counted, budget=budget, goal=goal)

