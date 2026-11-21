"""step17.py: choose a good build for a fixed scouting budget.

choose(blind_csv, label, budget=30, goal=None) maps the client's two requests
onto the search as follows:
  - "Best strong penalty-taker, sending scouts to see only 20 players":
    choose(blind_csv, label, budget=20, goal="Penalties+")
  - "Best one under 24, in a league we already scout": the age and league
    limits are not part of the search. The caller first writes a blind csv
    holding only the under-24 players from known leagues, then calls choose
    on that file.

The search itself is step7.choose (explore across flag families, then exploit
by predicted score), reused unchanged. Standard library only.
"""
from step7 import choose as _search


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, benchmarking at most `budget` rows.

    label(i) returns the goals of row i as a dict and may be called at most
    `budget` times. If `goal` names one goal column (with or without its +/-
    suffix), only that goal is optimised.
    """
    return _search(blind_csv, label, budget=budget, goal=goal)

