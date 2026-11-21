# step19.py
import step9

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means more likely a glitch, a typo, or unlike anything in train_csv."
  return step9.odd(train_csv, rows)

