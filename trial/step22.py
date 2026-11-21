# step22.py
#!/usr/bin/env python3 -B
"""
step22.py: guess.  Guess a new customer's goals from his or her input
numbers, with a range for each goal.  The client's two questions are
answered by the goals NUMWEBPURCHASES+ (online) and NUMSTOREPURCHASES+
(in store), and by RESPONSE+ (whether the customer answers the campaign).
The work is step12.guess, which wraps step2.guess: the nearest training
customers vote, the estimate is their median, and low..high is their
10th..90th percentile band.  Nothing here is new code; this file fixes
the client-facing signature.
"""
import step12

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal, where low..high is
  the range the true value should fall in.  ROW has its goal columns removed."""
  return step12.guess(train_csv, row)

