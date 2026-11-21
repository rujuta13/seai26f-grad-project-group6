#!/usr/bin/env python3 -B
"step12.py: guess. Guess each goal of a new player, with a range we are fairly sure of."
from step2 import guess as _guess

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high).  ROW has its goal columns removed,
  either as a dict {column name: value} or a list in header order minus goals.
  The estimate is the median goal of the K nearest train rows.  LOW..HIGH is
  the 10th..90th percentile range, so a wide range means low confidence."""
  return _guess(str(train_csv), row)

