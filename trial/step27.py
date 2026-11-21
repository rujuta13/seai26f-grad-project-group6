#!/usr/bin/env python3 -B
"step27.py: choose. Find the best customers while test-mailing at most BUDGET people."
from step17 import choose as _choose

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row of BLIND_CSV, where every goal cell is '?'.
  LABEL(i) returns row i's goals as a dict, and may be called at most BUDGET
  times (more raises RuntimeError).  GOAL, if given, optimises that one goal
  only, e.g. goal="NUMSTOREPURCHASES+" to target store visits over web orders."""
  return _choose(blind_csv, label, budget=budget, goal=goal)

