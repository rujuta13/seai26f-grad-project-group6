# step30.py
"""
step30: race.  Two client asks, each against a standard rival, on the same
test rows and the same yardstick (ezr.wins of each method's one pick).

  Ask 1: does our targeting beat simply mailing the highest-income customers?
         Rival "income": the test row with the highest INCOME.

  Ask 2: does our response guesser beat an off-the-shelf model?
         "response_guess": step12's guess of RESPONSE+, the highest
                          predicted response wins.
         "response_knn":  stdlib k-nearest-neighbour regression of RESPONSE+
                          on the training rows (off-the-shelf stand-in).

Methods from step10.race are kept: ours, random, ours_guess, knn.
Reuses step2 (model), step10 (race, _blind, KNN_K), step12 (guess), and
ezr (Tbl, csv, wins, xdist).

step20's race is not reused here: it needs an OVR column, which the
customer table does not have.
"""
import heapq
import math
import ezr
import step2
import step10
import step12

RESP   = "RESPONSE+"   # the response goal the client asks about
INCOME = "INCOME"      # the simple rival's sort key


def _at(tbl, name: str) -> int:
  "Column index of NAME in TBL, with or without its goal sign."
  for key in (name, name.rstrip("+-!")):
    if key in tbl.names: return tbl.names.index(key)
  raise ValueError(f"race: test_csv has no {name} column")


def _best(vals: list) -> int:
  "Index of the largest finite number in VALS; anything else counts as lowest."
  def key(i):
    v = vals[i]
    ok = isinstance(v, (int, float)) and math.isfinite(v)
    return v if ok else float("-inf")
  return max(range(len(vals)), key=key)


def _response_guess(train_csv, test, goals, at) -> int:
  "Our response guesser: pick the row with the highest predicted RESPONSE."
  name  = test.names[at]
  preds = [step12.guess(train_csv, step10._blind(r, goals))[name][0]
           for r in test.rows]
  return _best(preds)


def _response_knn(train_tbl, test, goals, at) -> int:
  "Off-the-shelf stand-in: kNN regression of RESPONSE, pick the highest."
  def predict(row) -> float:
    near = heapq.nsmallest(step10.KNN_K, train_tbl.rows,
                           key=lambda t: ezr.xdist(train_tbl, row, t))
    vs = [t[at] for t in near
          if isinstance(t[at], (int, float)) and math.isfinite(t[at])]
    return sum(vs) / len(vs) if vs else train_tbl.cols[at].mu
  preds = [predict(step10._blind(r, goals)) for r in test.rows]
  return _best(preds)


def race(train_csv, test_csv) -> dict[str, float]:
  "Each method's pick on TEST_CSV, scored by ezr.wins. Higher is better. Includes 'ours'."
  scores = step10.race(train_csv, test_csv)     # ours, random, ours_guess, knn

  test = ezr.Tbl(ezr.csv(test_csv))
  if not test.rows:
    raise ValueError("race: test_csv has no rows")
  goals = {c.at for c in test.y}
  score = ezr.wins(test)
  train_tbl, _ = step2._model(train_csv)

  resp_at = _at(test, RESP)
  inc_at  = _at(test, INCOME)

  picks = {
    "income":         _best([r[inc_at] for r in test.rows]),
    "response_guess": _response_guess(train_csv, test, goals, resp_at),
    "response_knn":   _response_knn(train_tbl, test, goals, resp_at),
  }
  for m, i in picks.items():
    scores[m] = float(score(test.rows[i]))

  if "ours" not in scores:
    raise RuntimeError("race: step10.race returned no 'ours' score")
  return scores

