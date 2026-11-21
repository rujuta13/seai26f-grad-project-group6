# step25.py
#!/usr/bin/env python3 -B
"""
step25.py: blame.  Why do some customers never respond to campaigns, and
what do the customers who complain have in common?
A customer is bad if its ydist (all goals together: RESPONSE+, COMPLAIN-,
and the purchase and visit goals) is in the worst quarter of the training
customers with every goal known.  Input columns are ranked by how much of
that bad/good split they explain, with identifier-like columns discounted.
Nothing new is needed, so this file delegates to step15.blame.
"""
import step15

def blame(train_csv: str) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  return step15.blame(train_csv)

