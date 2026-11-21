"""Step 17: choose. Find the best player for the client while scouting only a limited number of players.

Serves the client's requests:
  - "Find the best strong penalty-taker while sending scouts to see only 20 players."
  - "Find the best one under 24, in a league we already scout."

The search itself is step7.choose: explore the spread of the candidate pool,
then predict and probe one candidate at a time. This module adds a hard guard
on the label budget and the goal selection the client needs.

Constraints such as "under 24" or "in a league we already scout" are not
applied here. The caller passes a blind csv that already holds only the
eligible players, and the returned index refers to a row of that file.
"""

import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, labelling at most `budget` rows.

    blind_csv: MOOT csv whose goal cells are '?'. Only its input columns are
        used, and it should already contain only the eligible candidates.
    label(i): returns the goals of row i as a dict {goal: value}. It is called
        at most `budget` times; a call beyond that raises RuntimeError.
    goal: optional goal name (for example 'Penalties' or 'Penalties+'). When
        given, only that goal is optimised; otherwise all goals are summed.
    Returns the index of the best row found, ties going to the lower index.
    """
    calls = 0

    def counted(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError(f"label budget of {budget} exhausted")
        calls += 1
        return label(i)

    return step7.choose(blind_csv, counted, budget=budget, goal=goal)

