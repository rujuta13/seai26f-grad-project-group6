# step19.py
#!/usr/bin/env python3 -B
"""
step19.py: spot.  Flag player records that look like data-entry mistakes,
and players so unusual that their ratings deserve a second look.
The work is step9.odd: each row gets a suspicion score, the sum of how far
its inputs are from the nearest training row, how many input values were
never seen in training or fall outside the training range, and the worst
goal's z-score against the training goals.  Higher means odder.
Nothing here is new code; this file fixes the client-facing signature.
"""
import step9

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS, goals included; higher = more likely a glitch, typo, or unlike anything in TRAIN_CSV."
  return step9.odd(train_csv, rows)

