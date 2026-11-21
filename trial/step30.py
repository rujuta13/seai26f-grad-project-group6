# step30.py
"""Race for the client: does our targeting beat simply mailing the highest-income
customers, and does our response guesser beat an off-the-shelf model?

Targeting scores (higher is better) use wins: 100 at the best customer in
test_csv, 0 at an average one, on goals only. Rivals are step10's random
search and stdlib hill climber, plus "top_income", which labels the BUDGET
customers with the highest INCOME and keeps the best of them.

Response scores (higher is better) use skill: 100 when the guesses have zero
mean absolute error on RESPONSE, 0 when they are no better than always
guessing the training response rate. "response_ours" is step22.guess, and
"response_linear" is ridge regression on the numeric inputs, from step20.
"""
import ezr
import step2
import step4
import step7
import step10
import step20
import step22

INCOME = "INCOME"


def _at(tbl, name: str) -> int:
  "Column position of NAME in TBL's header."
  names = list(tbl.names)
  if name not in names:
    raise ValueError(f"race: no {name} column; have {names}")
  return names.index(name)


def _response_goal(tbl):
  "The RESPONSE goal column of TBL."
  hits = [g for g in step4._goals(tbl)
          if g.txt.rstrip("+-").upper() == "RESPONSE"]
  if not hits:
    raise ValueError(f"race: no RESPONSE goal in "
                     f"{[g.txt for g in step4._goals(tbl)]}")
  return hits[0]


def _top_income(tbl, cap: int) -> int:
  "The row a highest-income mailing picks: the CAP highest INCOME rows, best one kept."
  at = _at(tbl, INCOME)

  def key(i: int) -> float:
    v = tbl.rows[i][at]
    return -v if step2._known(v) else 1e30

  picks = sorted(range(len(tbl.rows)), key=key)[:cap]
  lab   = ezr.clone(tbl)
  for i in picks: ezr.addRow(lab, tbl.rows[i])
  goals = step4._goals(lab)            # goal stats learnt from the labels only
  return min(picks, key=lambda i: step7._cost(lab, tbl.rows[i], goals))


def _response(train_csv: str, test_csv: str) -> dict[str, float]:
  "Response guessers, scored by skill on the known RESPONSE cells of test_csv."
  ttbl = step2._load(train_csv)
  tbl  = step2._load(test_csv)
  resp = _response_goal(ttbl)
  goal_names = {g.txt for g in step4._goals(tbl)}

  known = [r[resp.at] for r in ttbl.rows if step2._known(r[resp.at])]
  base  = sum(known) / len(known)     # always guess the training rate

  lin   = step20._ridge(ttbl, resp)
  idx   = [i for i, r in enumerate(tbl.rows) if step2._known(r[resp.at])]
  truth = [tbl.rows[i][resp.at] for i in idx]
  ours, linear = [], []
  for i in idx:
    row = tbl.rows[i]
    q   = {name: row[at] for at, name in enumerate(tbl.names)
           if name not in goal_names}      # inputs only, goals hidden
    ours.append(step22.guess(train_csv, q)[resp.txt][0])
    linear.append(lin(row))
  return {"response_ours":   step20._skill(truth, ours, base),
          "response_linear": step20._skill(truth, linear, base)}


def race(train_csv, test_csv) -> dict[str, float]:
  "Scores on test_csv for ours and the rivals; higher is better for every key."
  out  = dict(step10.race(train_csv, test_csv))   # ours, random, stdlib_hillclimb
  tbl  = step2._load(test_csv)
  wins = step20._wins(tbl)
  cap  = min(step10.BUDGET, len(tbl.rows))
  out["top_income"] = wins(_top_income(tbl, cap))
  out.update(_response(train_csv, test_csv))
  return {k: float(v) for k, v in out.items()}

