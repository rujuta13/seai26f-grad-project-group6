# step27.py
#!/usr/bin/env python3 -B
"""
step27.py: choose.  Pick the best customers to target while test-mailing
at most BUDGET people.  The work is step7.choose: ezr.acquire labels
customers near the best so far and far from the rest, then returns the
best labelled row.  With GOAL set, only that goal counts.  The client's
two requests map onto the arguments as follows:
  - Best customers, test-mailing 50 people: budget=50.
  - Target store visits rather than web orders: goal="NUMSTOREPURCHASES+"
    (the "+" suffix is optional; step7 matches the bare name too).
Nothing here is new code; this file fixes the client-facing signature.
"""
import step7

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best customer found, labelling at most BUDGET rows via
  LABEL.  LABEL(i) returns the goals of row i as a dict.  If GOAL names
  one goal, only that goal is optimised."""
  return step7.choose(blind_csv, label, budget=budget, goal=goal)

