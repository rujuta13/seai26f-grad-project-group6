# step18.py
"""
step18: fix, for the client.  Two asks:
  1. "This player is close: what would he need to improve to make our
     shortlist?"  Call fix on his row (goals removed), with the inputs
     he cannot change (age, height, nation, and so on) in frozen.
  2. "What training focus would turn our weak penalty-taker into a
     reliable one?"  Call fix on that player's row, freezing every input
     except the skill columns that training can change.
Both reuse step8's fix unchanged.
"""
import step8


def fix(train_csv, row, frozen=()) -> dict:
  """The same row's inputs, with as few values changed as possible so that
  it is predicted good. ROW is a bad row with its goals removed. Columns
  named in FROZEN never change."""
  return step8.fix(train_csv, row, frozen=frozen)

