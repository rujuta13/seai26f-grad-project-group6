# step16.py
import step6

def explain(train_csv) -> str:
  "Reasons, in plain words a skeptical expert can check, for what makes rows good."
  return step6.explain(train_csv)

def why(train_csv, row) -> str:
  "One or two sentences: is ROW (goals included) good or bad, and why."
  return step6.why(train_csv, row)

