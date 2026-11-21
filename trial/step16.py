# step16.py
"""Justify for the client, in terms a manager or a video reviewer can check.

explain(): plain reasons for what makes rows good, from step6.explain.
why(): one or two sentences on one player, goals included, from step6.why.
Both are associations in the training data, not proven causes.
"""
import step6


def explain(train_csv) -> str:
  "Plain reasons, checkable by hand, for what makes rows good."
  return step6.explain(train_csv)


def why(train_csv, row) -> str:
  "One or two sentences: why this row (goals included) is good or bad."
  return step6.why(train_csv, row)

