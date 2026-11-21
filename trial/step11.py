# step11.py
"""
step11: the client's two asks.  families() sorts players into types (the
leaves of one tree over the inputs).  best() finds the players who are
strong and good from the spot: the best rows on all goals together.  The
goals are Strength+ and Penalties+, so "all goals together" is that pair.
Both are already built in step1, so this file reuses them unchanged.
"""
import step1


def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  return step1.families(train_csv)


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  return step1.best(train_csv)

