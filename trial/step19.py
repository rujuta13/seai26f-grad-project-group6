#!/usr/bin/env python3 -B
"step19.py: spot. Which player records look like data-entry mistakes, or are so unusual we should look twice before trusting their ratings?"
from step9 import odd as _odd

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means more likely a glitch, a typo, or unlike anything in TRAIN_CSV."
  return _odd(str(train_csv), list(rows))

