# step17.py
import step7

# The client's two questions map onto the existing tuner:
#   "best strong penalty-taker, 20 scouted"  -> goal="Penalties", budget=20
#   "best under 24, in a scouted league"     -> the caller passes only the
#       rows that meet the age and league limits in blind_csv, then labels
#       them with budget=30 (or fewer). The filter is not part of choose.

def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best row of BLIND_CSV, labelling at most BUDGET rows with LABEL. GOAL, if given, names the one goal to optimise."
  return step7.choose(blind_csv, label, budget=budget, goal=goal)

