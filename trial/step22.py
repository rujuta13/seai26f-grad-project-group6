# step22.py
#!/usr/bin/env python3 -B
"Guess each goal of a new customer from the other inputs, with a range we are fairly sure of, via ezr."
import step12

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Estimate each goal of ROW (a dict with its goal columns removed).
  Returns {goal: (estimate, low, high)} for every goal in TRAIN_CSV. The
  estimate is the median goal value of the nearest labelled training rows
  over the input columns; low..high is the 10th..90th percentile of those
  same neighbours, the range we are fairly sure the true value falls in.
  A wide range means the neighbours disagree."""
  out = {}
  for goal, (est, low, high) in step12.guess(train_csv, row).items():
    lo, hi = min(low, est, high), max(low, est, high)   # keep the order sane
    out[goal] = (float(est), float(lo), float(hi))
  return out

