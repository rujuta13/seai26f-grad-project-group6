#!/usr/bin/env python3 -B
"step21.py: remember. Group customers into the kinds of shopper they really are, and list the best customers: busy across channels, loyal, no complaints."
import step1

def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  return step1.families(str(train_csv))

def best(train_csv) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  return step1.best(str(train_csv))

