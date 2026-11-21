# step29.py
#!/usr/bin/env python3 -B
"Flag customer records that look wrong (impossible ages, odd incomes, typos) or behave unlike anyone else, via ezr. A high score means unusual, not proven fraud: a score cannot tell a mistake from a genuine outlier."
import step19

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS, goals included: higher means more likely
  a glitch, a typo, or unlike anything in TRAIN_CSV. Each row's score is the largest of
  its distance to the nearest training row over the inputs, and its most unusual cell
  in any input or goal column (an impossible age or income far from the training mean,
  a category never seen in training, or text or nan/inf where a number belongs).
  Unknown cells ('?') never raise a score by themselves. Scores say how unusual a row
  is, not whether it is fraud; check the top-scoring rows by hand."""
  return step19.odd(train_csv, rows)

