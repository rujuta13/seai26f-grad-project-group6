# step21.py
"""
step21: remember, for the client.  Two asks: group the 2,200 customers
into the kinds of shopper they really are, and find the best customers
(busy across channels, loyal, no complaints).  Both are already built in
step1, so this file reuses them unchanged.
"""
import step1


def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  return step1.families(train_csv)


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  return step1.best(train_csv)

