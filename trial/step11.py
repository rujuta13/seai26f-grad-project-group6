# step11.py
#!/usr/bin/env python3 -B
"""
step11.py: remember.  Families of players, and the best of them.
Both answers come from step1, which already does this on any MOOT csv:
families() grows a tree on the rows and labels each row by its leaf;
best() keeps the rows with the lowest ydist (all goals together), at most 5%.
Nothing here is new code; the player data just supplies the table.
"""
import step1

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  return step1.families(train_csv)

def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows, best on all goals together: at most 5% of rows."
  return step1.best(train_csv)

