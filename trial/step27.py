"""Step 27: choose. Find the best customers to target while test-mailing a limited number of people.

Serves the client's requests:
  - "Find the best customers to target while test-mailing only 50 people."
  - "If I care most about store visits rather than web orders, who should I target instead?"

The search itself is step7.choose: explore the spread of the candidate pool,
then predict and probe one candidate at a time. This module adds a hard guard
on the label budget and the goal selection the client needs. Passing a goal
name such as 'NUMSTOREPURCHASES+' (or the bare name 'NUMSTOREPURCHASES')
optimises only that goal, so store visits can be preferred over web orders.

The caller passes a blind csv whose goal cells are '?'. Only its input
columns are used, and the returned index refers to a row of that file.
"""

import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, labelling at most `budget` rows.

    blind_csv: MOOT csv whose goal cells are '?'. Only its input columns are used.
    label(i): returns the goals of row i as a dict {goal: value}. It is called
        at most `budget` times; a call beyond that raises RuntimeError.
    goal: optional goal name (for example 'NUMSTOREPURCHASES+' or
        'NUMSTOREPURCHASES'). When given, only that goal is optimised;
        otherwise all goals are summed.
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

