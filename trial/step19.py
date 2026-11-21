# step19.py
"""
step19: spot.  Flag player records that look like data-entry mistakes, and
players so unusual that their ratings deserve a second look.  Both asks are
step9's odd score, so this file reuses it unchanged.
"""
import step9


def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS (goals included): higher means more
  likely a glitch, a typo, or unlike anything in TRAIN_CSV."""
  return step9.odd(train_csv, rows)

