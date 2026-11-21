"""step17.py: choose a strong build while scouting only a fixed number of players.

Standard library only (Python 3.13). Reuses step7.choose (the two-phase
budgeted search) and step11.families (clustering that stays fast on files of
about 17,000 rows, where step1.families is too slow).

Client questions:
  - "Best strong penalty-taker, sending scouts to see only 20 players":
    choose(blind, label, budget=20, goal="Penalties+").
  - "Best one under 24, in a league we already scout": choose has no filter
    argument, so restrict blind_csv to the under-24 rows of that league in
    the caller before calling choose.
"""
from __future__ import annotations

import step7
from step11 import families as _scalable_families


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best build found within `budget` calls to label(i).

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column name: value}.
    budget: at most this many label calls; more raises RuntimeError.
    goal: optimise only this goal (name with or without +/- suffix).
          None optimises all goals together.
    """
    # step7.choose looks up `families` as a module global at call time, so
    # swap in the scalable clustering for the duration of the call. For files
    # of at most step11._FIT_CAP rows, step11.families delegates to
    # step1.families, so the result is the same as before.
    original = step7.families
    step7.families = _scalable_families
    try:
        return step7.choose(blind_csv, label, budget=budget, goal=goal)
    finally:
        step7.families = original

