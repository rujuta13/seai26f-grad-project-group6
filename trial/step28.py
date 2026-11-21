# step28.py
"""
step28: fix, for the client.  Two asks:
  1. "This customer is drifting away: what's the smallest nudge that
     brings them back?"  Call fix on the customer's row (goals removed),
     with the inputs the company cannot change (age, education, marital
     status, and so on) in frozen.
  2. "Which product offer would turn a deal-only shopper into a regular
     one?"  Call fix on that shopper's row, freezing every input except
     the product-spend columns that an offer can move.
Both reuse step8's fix unchanged.
"""
import step8


def fix(train_csv, row, frozen=()) -> dict:
  """The same row's inputs, with as few values changed as possible so that
  it is predicted good. ROW is a bad row with its goals removed. Columns
  named in FROZEN never change."""
  return step8.fix(train_csv, row, frozen=frozen)

