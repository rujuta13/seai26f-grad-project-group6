# step24.py
import step4
import step14

def stream(train_csv, rows) -> list[bool]:
  "For each row, as it arrives: True if it is worth a welcome offer (good on the goals), using only train_csv and that row."
  out = []
  for row in rows:                                    # one row at a time, in order
    out += step14.stream(train_csv, [row])            # goals estimated from train only
  return out

def drift(train_csv, rows) -> int | None:
  "Index of the first row where the recent new rows stop looking like train_csv, or None if they never do."
  return step4.drift(train_csv, rows)                 # checks only rows up to each index

