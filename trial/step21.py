# step21.py
#!/usr/bin/env python3 -B
"""
step21.py: remember.  Kinds of shopper, and the best customers.
families() groups customers into clusters by their tree leaf.  best()
keeps the rows with the lowest ydist on all goals together (channel
purchases, customer days, responses, and complaints), at most 5% of rows.
Both delegate to step11, which delegates to step1; no new logic is needed.
"""
import step11

def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  return step11.families(train_csv)

def best(train_csv) -> list[int]:
  "Row indices of the unusually good customers, best on all goals together: at most 5% of rows."
  return step11.best(train_csv)

