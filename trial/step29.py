#!/usr/bin/env python3 -B
"step29.py: spot. Flag customer records that look wrong (impossible ages, odd incomes, typos), and customers unlike anyone else: fraud, or just unusual?"
from step9 import odd as _odd

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means more likely a glitch, a typo, or unlike anything in TRAIN_CSV."
  return _odd(str(train_csv), list(rows))

