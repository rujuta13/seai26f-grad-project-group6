#!/usr/bin/env python3 -B
"step17.py: choose. Find the best flag set while labelling at most BUDGET builds."
from step7 import choose as _choose

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row of BLIND_CSV, where every goal cell is '?'.
  LABEL(i) returns row i's goals as a dict, and may be called at most BUDGET
  times (more raises RuntimeError).  GOAL, if given, optimises that one goal only."""
  if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
    raise ValueError("budget must be an integer of at least 1")
  return _choose(str(blind_csv), label, budget=budget, goal=goal)

