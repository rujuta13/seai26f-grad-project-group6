# step20.py
import ezr
import step10

def _top_rating(test: ezr.TBL) -> int:
  "Index of the row with the highest OVR, the simple 'top overall rating' list. Goals are never read."
  if "OVR" not in test.names: raise ValueError("race: no OVR column to rank by")
  at   = test.names.index("OVR")
  vals = [(r[at], i) for i, r in enumerate(test.rows) if r[at] != "?"]
  if not vals: raise ValueError("race: no OVR values to rank by")
  return max(vals, key=lambda t: (t[0], -t[1]))[1]   # ties: earlier row wins

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Each method picks one row of TEST_CSV; score it by wins on test (100 = best, 0 = average). Includes 'ours' and the 'top' overall-rating list."
  test   = ezr.Tbl(ezr.csv(test_csv))
  if not test.rows: raise ValueError(f"no rows in {test_csv}")
  scores = step10.race(train_csv, test_csv)          # ours, random, grid, linear
  score  = ezr.wins(test)
  scores["top"] = float(score(test.rows[_top_rating(test)]))
  return scores

