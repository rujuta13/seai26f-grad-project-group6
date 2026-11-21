# step29.py
import step9

# The client's two questions map onto step9's odd score:
#   "flag records that look wrong: impossible ages, odd incomes, typos"
#       -> a high score from a symbol never seen in train, text in a number
#          column, a missing goal, or a number far outside train's spread
#   "which customers behave unlike anyone else?"
#       -> a high score from a large distance to the nearest train row
# A high score means unusual or suspect, not fraud. A person must check the
# flagged rows before drawing that conclusion.

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means more likely a glitch, a typo, or unlike anything in train_csv."
  return step9.odd(train_csv, rows)

