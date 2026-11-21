# step29.py
"""
step29: spot, for the client.  Two asks: flag customer records that look
wrong (impossible ages, odd incomes, typos), and find customers who behave
unlike anyone else (fraud, or just unusual).  Both are step9's odd score,
so this file reuses it unchanged.
"""
import step9


def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS (goals included): higher means more
  likely a glitch, a typo, or unlike anything in TRAIN_CSV."""
  return step9.odd(train_csv, rows)

