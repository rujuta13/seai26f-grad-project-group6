# step19.py
"""Spot: flag player records that look like data-entry mistakes, or are so
unusual that their ratings deserve a second look.

odd() is step9.odd. Each row gets a suspicion score against TRAIN_CSV:
a text cell where a number belongs, a goal far from the training median
or from its nearest flag sets, or flags unlike any seen. A score near 1
is typical; several times that is worth checking. Rows may include their
goal cells (dict or full-width list), and missing cells add nothing.
"""
import step9


def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means a likelier glitch, typo, or unlike anything in TRAIN_CSV."
  rows   = list(rows)
  scores = [float(s) for s in step9.odd(train_csv, rows)]
  if len(scores) != len(rows):
    raise ValueError(f"odd: expected {len(rows)} scores, got {len(scores)}")
  return scores

