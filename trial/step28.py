#!/usr/bin/env python3 -B
"step28.py: fix. The smallest nudge that brings a drifting customer back, and the offer that turns a deal-only shopper into a regular one."
from step8 import fix as _fix

def fix(train_csv, row, frozen=()) -> dict:
  """The same row's inputs, with as few values changed as possible so that
  its predicted goals are good.  ROW has its goals removed, as a dict or a
  list in header order minus goals.  Columns named in FROZEN never change.
  Goals are not inputs, so they are never returned or changed."""
  return _fix(str(train_csv), row, frozen)

