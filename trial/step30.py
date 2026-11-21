# step30.py
import ezr
import step4
import step8
import step10

def _income(test: ezr.TBL) -> int:
  "Index of the row with the highest INCOME: the 'mail the richest' rule. Goals are never read."
  if "INCOME" not in test.names: raise ValueError("race: no INCOME column to rank by")
  at   = test.names.index("INCOME")
  vals = [(r[at], i) for i, r in enumerate(test.rows) if r[at] != "?"]
  if not vals: raise ValueError("race: no INCOME values to rank by")
  return max(vals, key=lambda t: (t[0], -t[1]))[1]   # ties: earlier row wins

def _guesser(train_csv: str, test: ezr.TBL) -> int:
  "Index of the test row whose goals the response guesser (step2.guess) estimates best. No labels used."
  tbl, score = step4._scorer(train_csv)
  goal_at    = {c.at for c in tbl.y}
  best, pick = None, 0
  for i, r in enumerate(test.rows):
    full = ["?" if at in goal_at else v for at, v in enumerate(r)]   # hide the goals
    s    = step8._est_score(train_csv, tbl, score, full)
    if best is None or s > best: best, pick = s, i                  # ties: earlier row wins
  return pick

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Each method picks one row of TEST_CSV; score it by wins on test (100 = best, 0 = average). 'ours' is the targeting tuner; 'income' mails the highest earners; 'guesser' and 'linear' are the response-guessing rivals."
  test = ezr.Tbl(ezr.csv(test_csv))
  if not test.rows: raise ValueError(f"no rows in {test_csv}")
  scores = step10.race(train_csv, test_csv)          # ours, random, grid, linear
  score  = ezr.wins(test)
  scores["income"]  = float(score(test.rows[_income(test)]))
  scores["guesser"] = float(score(test.rows[_guesser(train_csv, test)]))
  return scores

