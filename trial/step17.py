# step17.py
"""
step17: choose, for the client.  Two asks:
  1. The best strong penalty-taker, with scouts seeing only 20 players.
     "Strong penalty-taker" needs both goals, Strength+ and Penalties+,
     so leave goal=None and pass budget=20.
  2. The best player under 24 in a league we already scout.  Age and
     League are filters, not goals, so this signature cannot express
     them.  The caller narrows blind_csv to the eligible rows first
     (keep only rows with Age < 24 and League in the scouted leagues,
     with the header line kept), then calls choose on the narrowed file.
     The returned index is then a row index of that narrowed file.
Reuses step7.choose unchanged.
"""
import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row in BLIND_CSV, labelling at most BUDGET rows.
  BLIND_CSV has every goal cell '?'. LABEL(i) returns the goals of row i
  as a dict, and may be called at most BUDGET times (more raises an error).
  GOAL, if given, names the only goal to optimise; otherwise all goals count.

  Client ask 1: choose(path, label, budget=20)   # strong: Strength+ and Penalties+
  Client ask 2: narrow path to Age < 24 and scouted leagues first, then
                choose(narrowed_path, label, budget=30)."""
  return step7.choose(blind_csv, label, budget=budget, goal=goal)

