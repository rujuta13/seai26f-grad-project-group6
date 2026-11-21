# step28.py
import step8

# The client's two questions map onto step8's fix:
#   "drifting customer, smallest nudge back"  -> fix(train_csv, row)
#   "deal-only shopper into a regular one"    -> fix(train_csv, row, frozen=(...))
# FROZEN names the input columns that must not change, such as customer
# demographics. Goals are never changed, since they are estimated, not set.

def fix(train_csv, row, frozen=()) -> dict:
  "Inputs of ROW, changed as few as possible so that it becomes good. Columns named in FROZEN never change."
  return step8.fix(train_csv, row, frozen)

