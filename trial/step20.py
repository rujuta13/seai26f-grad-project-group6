#!/usr/bin/env python3 -B
"step20.py: race. Does our shortlist beat a plain top-OVR list, and does our penalty guesser beat an off-the-shelf model?"
import os, random, tempfile
from ezr import Num, Sym, Tbl, csv, the, wins
from step2 import guess
from step7 import choose
from step10 import BUDGET, _oracle, _random, _hill, _score_guess, _solve

PEN    = "penalties"  # the goal the penalty guesser must get right
RIDGE  = 1.0          # ridge penalty for the off-the-shelf model
LEVELS = 12           # Sym inputs with more levels than this are dropped

def _fit_one(tr, goal, lam=RIDGE):
  """Ridge regression of one GOAL on TR's inputs: Nums z-scored,
  few-valued Syms one-hot.  Returns predict(row) -> float."""
  cols   = [c for c in tr.x if type(c) is Num or len(c.has) <= LEVELS]
  levels = {c.at: sorted(c.has, key=str) for c in cols if type(c) is Sym}
  def vec(row) -> list[float]:
    x = [1.0]
    for c in cols:
      v = row[c.at]
      if type(c) is Sym:
        x += [1.0 if v == w else 0.0 for w in levels[c.at]]
      else:
        ok = isinstance(v, (int, float))
        x.append((v - c.mu) / (c.sd or 1.0) if ok else 0.0)
    return x
  d = len(vec(tr.rows[0]))
  A = [[0.0] * d for _ in range(d)]
  b = [0.0] * d
  for r in tr.rows:
    y = r[goal.at]
    if not isinstance(y, (int, float)): continue
    x = vec(r)
    for i in range(d):
      b[i] += x[i] * y
      for j in range(d): A[i][j] += x[i] * x[j]
  for i in range(d): A[i][i] += lam
  w = _solve(A, b)
  return lambda row: sum(p * q for p, q in zip(w, vec(row)))

def race(train_csv, test_csv) -> dict[str, float]:
  """Scores, higher is better, keyed by method.
  Shortlist (one pick from TEST_CSV, scored by wins: 100 = best row,
  0 = an average one): 'ours' (step7.choose), 'top_ovr' (the row with the
  highest OVR), 'random' and 'hillclimb' (same label budget, as in step10).
  Penalty guess (every TEST_CSV row's Penalties, scored by
  1/(1 + mean error in train sds)): 'penalty_ours' (step2.guess) and
  'penalty_linreg' (off-the-shelf ridge regression)."""
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

  # ours: step7.choose on a blind copy of TEST, every goal shown as '?'
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

  # a plain top-overall-rating list: the single row with the highest OVR
  ovr = [i for i, n in enumerate(names) if n.upper() == "OVR"]
  if not ovr: raise ValueError(f"no OVR column in {names}")
  def rating(i) -> float:
    v = rows[i][ovr[0]]
    return v if isinstance(v, (int, float)) else float("-inf")
  res["top_ovr"] = float(score(rows[max(range(len(rows)), key=rating)]))

  # standard rivals with the same label budget
  res["random"]    = float(score(rows[_random(test, _oracle(test, BUDGET),
                                              BUDGET, rng)]))
  res["hillclimb"] = float(score(rows[_hill(test, _oracle(test, BUDGET),
                                             BUDGET, rng)]))

  # penalty guesser: ours vs an off-the-shelf ridge regression, trained on TRAIN
  tr = Tbl(csv(str(train_csv)))
  pen = next((c for c in tr.y if c.txt.rstrip("+-").lower() == PEN), None)
  if pen is None: raise ValueError(f"no penalties goal in {tr.names}")
  pname = pen.txt
  sd    = {pname: pen.sd or 1.0}
  pat   = {pname: pen.at}

  ours = []
  for r in rows:
    x   = {n: v for n, v in zip(names, r) if n[-1] not in "+-"}
    est = guess(str(train_csv), x)
    ours.append({pname: est[pname][0]})
  res["penalty_ours"] = float(_score_guess(ours, rows, pat, sd))

  predict = _fit_one(tr, pen)
  res["penalty_linreg"] = float(_score_guess([{pname: predict(r)} for r in rows],
                                             rows, pat, sd))
  return res

