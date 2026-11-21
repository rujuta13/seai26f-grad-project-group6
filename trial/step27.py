# step27.py
import ezr
import step17

ALIAS = {                                # client words -> goal base names
  "store visits":    "numstorepurchases",
  "store":           "numstorepurchases",
  "store purchases": "numstorepurchases",
  "web orders":      "numwebpurchases",
  "web":             "numwebpurchases",
  "web purchases":   "numwebpurchases",
  "catalog orders":  "numcatalogpurchases",
}

def _goal_name(blind_csv, goal) -> str | None:
  "The header name of the one goal that GOAL refers to, or None if GOAL is None."
  if goal is None: return None
  rows = ezr.csv(blind_csv)
  hdr  = next(rows, None)
  if hdr is None: raise ValueError("choose: no rows to choose from")
  hdr   = [str(s) for s in hdr]
  goals = [s.rstrip("+-") for s in hdr if s[-1:] in ("+", "-")]
  if not goals: raise ValueError("choose: the file has no goal columns")
  g     = " ".join(str(goal).replace("_", " ").lower().split())
  base  = ALIAS.get(g, g.replace(" ", ""))
  for name in goals:
    if name.lower() == base: return name
  hits = [name for name in goals if base and base in name.lower()]
  if len(hits) == 1: return hits[0]
  if not hits: raise ValueError(f"choose: no goal named {goal!r}")
  raise ValueError(f"choose: {goal!r} matches more than one goal: "
                   f"{', '.join(hits)}")

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row in BLIND_CSV (goals all '?'), calling LABEL
  at most BUDGET times. If GOAL names one goal, optimise only that one.
  GOAL may be a column name or a client phrase such as 'store visits'."""
  return step17.choose(blind_csv, label, budget=budget,
                       goal=_goal_name(blind_csv, goal))

