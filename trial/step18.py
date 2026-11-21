# step18.py
#!/usr/bin/env python3 -B
"""
step18.py: fix.  Given a player who is not yet on the shortlist (goals
removed), find the fewest input attributes to change so that the player
becomes good.  The work is step8.fix: it tries one changed attribute, then
two, and so on, and judges each candidate by its predicted goals against
the keep cut from step4.  Attributes named in FROZEN are never changed,
so a coach can lock traits that are fixed, such as age or height.
Nothing here is new code; this file fixes the client-facing signature.
"""
import step8

def fix(train_csv: str, row: dict, frozen=()) -> dict:
  """Return ROW's inputs with as few values changed as possible so the
  player becomes good.  Columns named in FROZEN are never changed."""
  return step8.fix(train_csv, row, frozen=frozen)

