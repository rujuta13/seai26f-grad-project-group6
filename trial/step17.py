# step17.py
"""Choose: the best build found while labelling at most BUDGET builds.

choose() is the client-facing entry point for step7.choose. It sees only
the flag sets in blind_csv, where every goal cell is '?'. It buys labels
through label(i) and returns the index of the best build found. With
goal=..., only that one goal is optimised. The budget is enforced here as
well: a call to label beyond BUDGET raises an error.
"""
import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best build found, labelling at most BUDGET rows via LABEL."
  calls = [0]                          # labels bought so far

  def guarded(i: int) -> dict:
    "Pass the label through, refusing to pay past the budget."
    calls[0] += 1
    if calls[0] > budget:
      raise RuntimeError(f"choose: label budget of {budget} exceeded")
    return label(i)

  return step7.choose(blind_csv, guarded, budget=budget, goal=goal)

