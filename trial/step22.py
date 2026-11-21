# step22.py
"""
step22: guess, for the client.  Two asks: how much a new customer will buy
online versus in store, and whether they will answer the next campaign.
Both are goals in the customer table (NUMWEBPURCHASES+, NUMSTOREPURCHASES+,
RESPONSE+), so this is step12's guess with its goal interval, unchanged.
"""
import step12


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal, where low..high is
  the range we are fairly sure the true value falls in. ROW has its goal
  columns removed."""
  return step12.guess(train_csv, row)

