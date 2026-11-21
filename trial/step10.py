#!/usr/bin/env python3 -B
"step10.py: race. Our tuner and guesser against standard rivals, on the same test data."
import os, random, tempfile
from math import isfinite
from ezr import Sym, Tbl, clone, csv, the, wins, xdist, ydist
from step2 import guess
from step7 import choose

BUDGET = 30   # labels each tuner may spend on the test rows

def _oracle(tbl, budget):
  "A labeller for TBL's rows by index; it refuses to go past BUDGET asks."
  def label(i):
    if label.asks >= budget: raise RuntimeError(f"more than {budget} labels asked")
    label.asks += 1
    return tbl.rows[i]
  label.asks = 0
  return label

def _random(tbl, label, budget, rng) -> int:
  "Label BUDGET rows at random; keep the best of them."
  n     = len(tbl.rows)
  picks = rng.sample(range(n), min(budget, n))
  lab   = clone(tbl, [label(i) for i in picks])
  return min(picks, key=lambda i: ydist(lab, tbl.rows[i]))

def _hill(tbl, label, budget, rng) -> int:
  """A stdlib-only local search (the stdlib has no tuner): from a random
  start, label the unlabelled row nearest the current best in input space,
  and move to it if it scores better."""
  rows, n = tbl.rows, len(tbl.rows)
  best    = rng.randrange(n)
  seen    = {best: label(best)}
  while len(seen) < min(budget, n):
    near = min((j for j in range(n) if j not in seen),
               key=lambda j: xdist(tbl, rows[best], rows[j]))
    seen[near] = label(near)
    lab = clone(tbl, list(seen.values()))
    if ydist(lab, rows[near]) < ydist(lab, rows[best]): best = near
  return best

def _solve(A, b) -> list[float]:
  "Gauss-Jordan elimination with partial pivoting: the w in A.w = b."
  n = len(b)
  M = [row[:] + [b[i]] for i, row in enumerate(A)]
  for c in range(n):
    p = max(range(c, n), key=lambda r: abs(M[r][c]))
    M[c], M[p] = M[p], M[c]
    if abs(M[c][c]) < 1e-12: continue
    for r in range(n):
      if r != c and M[r][c]:
        f = M[r][c] / M[c][c]
        M[r] = [a - f * m for a, m in zip(M[r], M[c])]
  return [M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0
          for i in range(n)]

def _fit(tr, lam=1.0):
  """Ridge regression on TR's input columns, one model per goal.
  This is the off-the-shelf rival: Sym inputs one-hot, Num inputs z-scored."""
  levels = {c.at: sorted(c.has, key=str) for c in tr.x if type(c) is Sym}
  def vec(row) -> list[float]:
    x = [1.0]
    for c in tr.x:
      v = row[c.at]
      if type(c) is Sym:
        x += [1.0 if v == w else 0.0 for w in levels[c.at]]
      else:
        ok = isinstance(v, (int, float))
        x.append((v - c.mu) / (c.sd or 1.0) if ok else 0.0)
    return x
  X = [vec(r) for r in tr.rows]
  d = len(X[0])
  A = [[0.0] * d for _ in range(d)]
  for x in X:
    for i in range(d):
      for j in range(d):
        A[i][j] += x[i] * x[j]
  for i in range(d): A[i][i] += lam
  W = {}
  for g in tr.y:
    b = [0.0] * d
    for x, r in zip(X, tr.rows):
      if isinstance(r[g.at], (int, float)):
        for i in range(d): b[i] += x[i] * r[g.at]
    W[g.txt] = _solve(A, b)
  def predict(row) -> dict:
    x = vec(row)
    return {k: sum(a * b for a, b in zip(w, x)) for k, w in W.items()}
  return predict

def _score_guess(preds, rows, goal_at, sd) -> float:
  "1/(1 + mean goal error, in train sds). Higher is better; 0 if any guess is not finite."
  errs = []
  for pred, r in zip(preds, rows):
    for g, p in pred.items():
      actual = r[goal_at[g]]
      if actual == "?": continue
      if not isfinite(p): return 0.0
      errs.append(abs(p - actual) / sd[g])
  return 1 / (1 + sum(errs) / len(errs)) if errs else 0.0

def race(train_csv, test_csv) -> dict[str, float]:
  """Scores, higher is better, keyed by method.
  Tuners each pick one row of TEST_CSV, scored by wins (100 = the best row,
  0 = an average one).  Keys: 'ours' (step7.choose), 'random', and 'hillclimb'
  (stdlib has no tuner, so this is a stdlib-only local search).
  Guessers predict every goal of every TEST_CSV row from TRAIN_CSV, scored by
  1/(1 + mean goal error in train sds).  Keys: 'guess_ours' (step2.guess)
  and 'guess_linreg' (stdlib ridge regression)."""
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

  res["random"]    = float(score(rows[_random(test, _oracle(test, BUDGET),
                                              BUDGET, rng)]))
  res["hillclimb"] = float(score(rows[_hill(test, _oracle(test, BUDGET),
                                             BUDGET, rng)]))

  # guessers: predict each test row's goals from train alone
  tr = Tbl(csv(str(train_csv)))
  sd = {c.txt: c.sd or 1.0 for c in tr.y}
  ours = []
  for r in rows:
    x   = {n: v for n, v in zip(names, r) if n[-1] not in "+-"}
    est = guess(str(train_csv), x)
    ours.append({g: est[g][0] for g in goals if g in est})
  res["guess_ours"]   = _score_guess(ours, rows, goal_at, sd)
  predict             = _fit(tr)
  res["guess_linreg"] = _score_guess([predict(r) for r in rows],
                                     rows, goal_at, sd)
  return res

