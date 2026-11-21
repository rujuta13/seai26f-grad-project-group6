# step21.py
import step11

def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of train_csv, in file order."
  return step11.families(train_csv)

def best(train_csv) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  return step11.best(train_csv)

