# step20.py
"""
step20: race.  Our shortlist against a simple "top overall rating" list, and
our penalty guesser against an off-the-shelf model, all on the same test rows
and the same yardstick (ezr.wins of each method's one pick).

Methods, as returned by step10.race:
  ours        our tuner: step7.choose, labelling at most the.Stop rows.
  random      random search with the same label budget.
  ours_guess  our guesser: step4's predicted ydist, no labels.
  knn         off-the-shelf stand-in: k-nearest-neighbour regression.
Added here:
  overall     the "top overall rating" list: the test row with the highest
              OVR. OVR is an input column, so no labels are needed.

Reuses step10 (race) and ezr (Tbl, csv, wins).
"""
import ezr
import step10


def _overall(test) -> int:
  "Index of the test row with the highest OVR; non-numeric OVR counts as lowest."
  if "OVR" not in test.names:
    raise ValueError("race: test_csv has no OVR column")
  at = test.names.index("OVR")

  def key(i):
    v = test.rows[i][at]
    return v if isinstance(v, (int, float)) else float("-inf")

  return max(range(len(test.rows)), key=key)


def race(train_csv, test_csv) -> dict[str, float]:
  "Each method's pick on TEST_CSV, scored by ezr.wins. Higher is better. Includes 'ours'."
  scores = step10.race(train_csv, test_csv)
  test   = ezr.Tbl(ezr.csv(test_csv))
  score  = ezr.wins(test)
  scores["overall"] = float(score(test.rows[_overall(test)]))
  if "ours" not in scores:
    raise RuntimeError("race: step10.race returned no 'ours' score")
  return scores

