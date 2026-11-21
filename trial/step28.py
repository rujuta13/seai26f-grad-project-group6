# step28.py
"""Nudge: the smallest change to a drifting customer's inputs that brings them back to good.

fix() is step18.fix, which is step8.fix plus a check that no frozen column
moved. It searches only the input (x) columns, judges each candidate with
step2.guess and step4's best-5% cutoff, and never changes a column named
in frozen.
"""
import step18


def fix(train_csv, row, frozen=()) -> dict:
  "Inputs of ROW with as few values changed as possible so it is good; FROZEN never change."
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = tuple(frozen)
  return dict(step18.fix(train_csv, row, frozen=frozen))

