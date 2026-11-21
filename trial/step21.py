# step21.py
"""Customer families and best customers, for the client's two questions.

families(): the kinds of shopper, as k-medoids families over the input
columns. It reuses step11.families, which samples medoid candidates so it
stays fast on large tables, and 2,200 customers is well within its reach.

best(): the best customers, meaning busy across channels, loyal, and with
no complaints, as measured by the goals. It reuses step1.best, which takes
the best 5% of rows on all goals together.
"""
import step1
import step11


def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of train_csv, in file order."
  return [int(k) for k in step11.families(train_csv)]


def best(train_csv) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  return [int(i) for i in step1.best(train_csv)]

