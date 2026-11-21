# step18.py
#!/usr/bin/env python3 -B
"Fewest input changes that make a bad build good, leaving frozen columns alone, via ezr."
import step8

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of ROW (goals removed) with as few values changed as possible so that
  it becomes good. Columns named in FROZEN never change. Raises ValueError if no
  fix is found, or if FROZEN names a column the table does not have."""
  return step8.fix(train_csv, row, frozen)

