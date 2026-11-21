# step29.py
#!/usr/bin/env python3 -B
"""
step29.py: spot.  Flag customer records that look wrong (impossible ages,
odd incomes, typos), and customers who behave unlike anyone else.
The work is step9.odd: each row gets a suspicion score, the sum of how far
its inputs are from the nearest training row, how many input values were
never seen in training or fall outside the training range (so an age far
beyond anyone in training scores high), and the worst goal's z-score
against the training goals.  Higher means odder.  Missing cells add
nothing.  Nothing new is needed, so this file delegates to step9.
"""
import step9

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS, goals included; higher = more likely a glitch, typo, or unlike anything in TRAIN_CSV."
  return step9.odd(train_csv, rows)

