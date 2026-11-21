# step12.py
#!/usr/bin/env python3 -B
"""
step12.py: guess.  Guess the goals of one player from his input numbers,
with a range for each goal.  The work is step2.guess: the nearest training
rows vote, ESTIMATE is their median, and LOW..HIGH is their 10th..90th
percentile, an 80% band.  A narrow band means the nearest players agree;
a wide one means they do not, so the range shows how confident the guess is.
"""
import step2

def guess(train_csv: str, row: dict) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal, where LOW..HIGH is
  the range the true value should fall in (an 80% band from the nearest
  training rows).  ROW has its goal columns removed."""
  out = step2.guess(train_csv, row)
  return {g: (float(est), float(min(lo, est)), float(max(hi, est)))
          for g, (est, lo, hi) in out.items()}

