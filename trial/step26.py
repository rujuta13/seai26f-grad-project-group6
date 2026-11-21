# step26.py
"""Justify for the client: why a group should get the next campaign, and why
a customer is marked low value, in reasons the client can act on.

explain(): plain reasons for what makes rows good, from step6.explain.
why(): one or two sentences on one customer, goals included, from step6.why.
Both are associations in the training data, not proven causes.
"""
import step6


def explain(train_csv) -> str:
  "Reasons, in plain words a skeptical expert can check, for what makes rows good."
  return str(step6.explain(train_csv))


def why(train_csv, row) -> str:
  "One or two sentences on why this row (goals included) is good or bad."
  return str(step6.why(train_csv, row))

