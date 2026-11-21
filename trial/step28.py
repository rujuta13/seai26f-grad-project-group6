# step28.py
"Step 28: fix, for the client. The fewest input changes that turn a drifting customer back into a good one, leaving frozen inputs alone."
import step18

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of bad ROW with as few values changed as possible so that it
  becomes good. ROW has its goal columns removed. Columns named in FROZEN
  never change. Returns {input column name: value}."""
  return step18.fix(train_csv, row, frozen)

