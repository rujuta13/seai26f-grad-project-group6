#!/usr/bin/env python3 -B
"step24.py: flow. Keep the good new sign-ups as they arrive; warn when this month's sign-ups stop looking like our usual customers."
from step4 import stream as _stream, drift as _drift

def stream(train_csv, rows) -> list[bool]:
  """One keep (True) or drop (False) per row of ROWS, in order.  A row is kept
  if its goals are at least as good as the best rows of TRAIN_CSV.  Each
  decision uses only TRAIN_CSV and the row in hand, so it is causal."""
  return _stream(str(train_csv), rows)

def drift(train_csv, rows) -> int | None:
  """Index into ROWS of the first row at which the last W rows' distances to
  TRAIN_CSV stop looking like TRAIN_CSV's own spacing, or None if they never
  do.  Reads ROWS in order and stops at the first warning."""
  return _drift(str(train_csv), rows)

