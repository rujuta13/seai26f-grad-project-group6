# step27.py
import step7

# The client's two questions map onto step7's tuner:
#   "best customers, test-mailing only 50 people"
#       -> choose(blind_csv, label, budget=50)
#   "if I care most about store visits rather than web orders"
#       -> choose(blind_csv, label, budget=50, goal="NUMSTOREPURCHASES")
# GOAL names one goal, with or without its + or - suffix; the other goals are ignored.
# LABEL(i) returns the goals of row i as a dict keyed by the full header names.

def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best row of BLIND_CSV, labelling at most BUDGET rows with LABEL. GOAL, if given, names the one goal to optimise."
  return step7.choose(blind_csv, label, budget=budget, goal=goal)

