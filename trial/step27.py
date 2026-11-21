# step27.py
"""Choose for the client: the best customer to target, with a fixed mailing budget.

choose() is step17.choose, the client-facing entry point for step7.choose.
It sees only the customer inputs in blind_csv, where every goal cell is '?'.
It buys labels through label(i), at most budget of them, and returns the
index of the best customer found. With goal=..., only that one goal is
optimised. The client can name a goal in plain words, such as "store visits"
or "web orders", and those words are translated to goal column names here.
"""
import step17

# plain words the client may use, mapped to goal column names
GOAL_WORDS = {
  "store":        "NUMSTOREPURCHASES",
  "store visits": "NUMSTOREPURCHASES",
  "web":          "NUMWEBPURCHASES",
  "web orders":   "NUMWEBPURCHASES",
}


def _goal_name(goal):
  "The goal column name for GOAL, translating plain words; None stays None."
  if goal is None: return None
  return GOAL_WORDS.get(str(goal).strip().lower(), goal)


def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best customer found, labelling at most BUDGET rows via LABEL."
  return int(step17.choose(blind_csv, label, budget=budget,
                           goal=_goal_name(goal)))

