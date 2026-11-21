# step27.py
"""
step27: choose, for the client.  Two asks:
  1. The best customers to target while test-mailing only 50 people.
     Both goals count, so leave goal=None and pass budget=50.
  2. Target store visits rather than web orders.  goal="NUMSTOREPURCHASES+"
     names the only goal to optimise, so the web-purchase goal is ignored.
Reuses step7.choose unchanged.
"""
import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row in BLIND_CSV, labelling at most BUDGET rows.
  BLIND_CSV has every goal cell '?'. LABEL(i) returns the goals of row i
  as a dict, and may be called at most BUDGET times (more raises an error).
  GOAL, if given, names the only goal to optimise; otherwise all goals count.

  Client ask 1: choose(path, label, budget=50)
  Client ask 2: choose(path, label, budget=50, goal="NUMSTOREPURCHASES+")"""
  return step7.choose(blind_csv, label, budget=budget, goal=goal)

