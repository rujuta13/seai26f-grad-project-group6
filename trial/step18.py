# step18.py
import step8

def fix(train_csv, row, frozen=()) -> dict:
  "Inputs of ROW, changed as few as possible so that it becomes good. Columns named in FROZEN never change."
  return step8.fix(train_csv, row, frozen)

