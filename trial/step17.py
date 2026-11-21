# step17.py
#!/usr/bin/env python3 -B
"""
step17.py: choose.  Scouts can watch only BUDGET players, so choose which
ones to label.  The work is step7.choose: ezr.acquire labels players near
the best so far and far from the rest, then returns the best labelled row.
Nothing here is new code; this file fixes the client-facing signature.

The client's two requests use this function as follows:
  - Best strong penalty-taker, 20 scouted players: goal="Penalties+", budget=20.
  - Best player under 24 in a league we already scout: the caller writes a
    blind_csv that holds only that league's players aged under 24, then calls
    choose() on it.  The function does not filter by age or league itself.
"""
import step7

def choose(blind_csv: str, label, budget: int = 30,
           goal: str | None = None) -> int:
  "Index of the best row found, labelling at most BUDGET rows via LABEL."
  return step7.choose(blind_csv, label, budget=budget, goal=goal)

