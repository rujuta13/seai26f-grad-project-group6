# step28.py
#!/usr/bin/env python3 -B
"""
step28.py: fix.  Given a customer who is drifting away (goals removed),
find the fewest input columns to change so that the customer becomes good.
The work is step8.fix: it tries one changed column, then two, and so on,
and judges each candidate by its predicted goals against the keep cut from
step4.  Columns named in FROZEN are never changed, so a fixed trait such as
age or marital status can be locked.  For the offer question, freeze every
column the offer cannot touch and let the search choose among the rest.
Nothing here is new code; this file fixes the client-facing signature.
"""
import step8

def fix(train_csv, row, frozen=()) -> dict:
  """Return ROW's inputs with as few values changed as possible so the
  customer becomes good.  Columns named in FROZEN are never changed."""
  return step8.fix(train_csv, row, frozen=frozen)

