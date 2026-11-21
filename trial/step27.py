# step27.py
#!/usr/bin/env python3 -B
"Choose the best customer to target while labelling at most BUDGET customers, optionally for one goal only, via ezr."
import step17

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best customer found, labelling at most BUDGET rows of BLIND_CSV.
  BLIND_CSV has every goal cell '?'. LABEL(i) returns the goals of row i as a
  {goal name: value} dict and may be called at most BUDGET times; a call past
  that raises RuntimeError. If GOAL names one goal (with or without its +/-
  suffix, ignoring case), only that goal counts; otherwise all goals count.
  For a test mailing of 50 people, pass budget=50. For store visits rather
  than web orders, pass goal="NUMSTOREPURCHASES"."""
  return step17.choose(blind_csv, label, budget=budget, goal=goal)

