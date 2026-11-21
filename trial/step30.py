#!/usr/bin/env python3 -B
"step30.py: race. Does our targeting beat simply mailing the highest-income customers, and does our response guesser beat an off-the-shelf model?"
import os, random, tempfile
from ezr import Tbl, csv, the, wins
from step7 import choose
from step10 import BUDGET, _oracle, _random, _hill, _score_guess
from step20 import _fit_one
from step22 import guess

GOAL   = "RESPONSE"  # the goal the response guesser must get right
INCOME = "INCOME"    # the input the simple rival ranks customers by

def _col(names, want: str) -> int:
  "Index of the column named WANT, ignoring case and any goal sign."
  hits = [i for i, n in enumerate(names) if n.rstrip("+-").upper() == want]
  if not hits: raise ValueError(f"no {want} column in {names}")
  return hits[0]

def race(train_csv, test_csv) -> dict[str, float]:
  """Scores, higher is better, keyed by method.
  Targeting (one pick from TEST_CSV, scored by wins: 100 = best row,
  0 = an average one): 'ours' (step7.choose, same label budget as the
  rivals), 'top_income' (the customer with the highest INCOME), 'random'
  and 'hillclimb' (same label budget, as in step10).
  Response guess (RESPONSE of every TEST_CSV row, scored by
  1/(1 + mean error in train sds)): 'guess_ours' (step22.guess) and
  'guess_linreg' (off-the-shelf ridge regression)."""
  random.seed(the.Seed)
  rng     = random.Random(the.Seed)
  raw     = list(csv(str(test_csv)))
  if len(raw) < 2: raise ValueError(f"no rows in {test_csv}")
  names   = raw[0]
  test    = Tbl(raw)
  rows    = test.rows
  goals   = [n for n in names if n[-1] in "+-"]
  goal_at = {n: names.index(n) for n in goals}
  score   = wins(test)
  res     = {}

  # ours: targeting by step7.choose on a blind copy of TEST, goals as '?'
  def ask(i) -> dict:
    return {g: rows[i][goal_at[g]] for g in goals}
  with tempfile.TemporaryDirectory() as d:
    blind = os.path.join(d, "blind.csv")
    with open(blind, "w", encoding="utf-8") as f:
      f.write(",".join(names) + "\n")
      for r in rows:
        f.write(",".join("?" if n in goal_at else str(v)
                         for n, v in zip(names, r)) + "\n")
    res["ours"] = float(score(rows[choose(blind, ask, budget=BUDGET)]))

  # the client's plain rival: mail the single highest-income customer
  inc = _col(names, INCOME)
  def income(i) -> float:
    v = rows[i][inc]
    return v if isinstance(v, (int, float)) else float("-inf")
  res["top_income"] = float(score(rows[max(range(len(rows)), key=income)]))

  # standard rivals with the same label budget
  res["random"]    = float(score(rows[_random(test, _oracle(test, BUDGET),
                                              BUDGET, rng)]))
  res["hillclimb"] = float(score(rows[_hill(test, _oracle(test, BUDGET),
                                             BUDGET, rng)]))

  # response guesser: ours vs an off-the-shelf ridge regression, trained on TRAIN
  tr   = Tbl(csv(str(train_csv)))
  resp = next((c for c in tr.y if c.txt.rstrip("+-").upper() == GOAL), None)
  if resp is None: raise ValueError(f"no {GOAL} goal in {tr.names}")
  pname = resp.txt
  sd    = {pname: resp.sd or 1.0}
  pat   = {pname: _col(names, GOAL)}

  ours = []
  for r in rows:
    x   = {n: v for n, v in zip(names, r) if n[-1] not in "+-"}
    est = guess(str(train_csv), x)
    ours.append({pname: est[pname][0]})
  res["guess_ours"] = float(_score_guess(ours, rows, pat, sd))

  predict = _fit_one(tr, resp)
  res["guess_linreg"] = float(_score_guess([{pname: predict(r)} for r in rows],
                                           rows, pat, sd))
  return res

