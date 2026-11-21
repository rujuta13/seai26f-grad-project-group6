# step25.py
#!/usr/bin/env python3 -B
"Input columns to blame for bad customers (bad on the goals, such as never responding or complaining), via ezr."
import step15

def blame(train_csv) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  A row is bad if its goals are in the worst quarter by goal distance (ydist).
  Each input column is scored by its largest lift: how much more often bad rows
  fall on one side of a candidate split than they fall in the table overall.
  All goal columns, including RESPONSE+ and COMPLAIN-, are judged together.
  Rows with an unknown goal cell are ignored. Ties keep file order."""
  return step15.blame(train_csv)

