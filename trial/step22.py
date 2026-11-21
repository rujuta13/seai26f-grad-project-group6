#!/usr/bin/env python3 -B
"step22.py: guess. Guess how much a new customer will buy online versus in store, and whether they will answer our next campaign, each with a range we are fairly sure of."
from step12 import guess as _guess

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high).  ROW has its goal columns removed,
  either as a dict {column name: value} or a list in header order minus goals.
  The client's questions use NUMWEBPURCHASES+ (online purchases),
  NUMSTOREPURCHASES+ (in-store purchases), and RESPONSE+ (answers the next
  campaign).  LOW..HIGH is the 10th..90th percentile range over the nearest
  train rows, so a wide range means low confidence."""
  return _guess(str(train_csv), row)

