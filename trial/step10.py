# step10.py
import os
import random
import tempfile
import ezr
import step2
import step7

BUDGET = 30    # labels each tuner may request
RIDGE  = 1e-6  # a tiny ridge keeps the off-the-shelf linear fit stable

def _num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _blind(test: ezr.TBL, rows: ezr.ROWS) -> str:
  "Write ROWS to a temp csv with goal cells blanked to '?'; return its path."
  goal_ats = {c.at for c in test.y if type(c) is ezr.Num}
  fd, path = tempfile.mkstemp(suffix=".csv")
  with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(",".join(test.names) + "\n")
    for r in rows:
      f.write(",".join("?" if at in goal_ats else str(v)
                       for at, v in enumerate(r)) + "\n")
  return path

def _mean(tbl: ezr.TBL, at: int) -> float:
  "Mean of the numeric cells in column AT of TBL; 0 if there are none."
  vals = [r[at] for r in tbl.rows if _num(r[at])]
  return sum(vals) / len(vals) if vals else 0.0

def _cell(v, m: float) -> float:
  "V if it is a number, else the column mean M."
  return v if _num(v) else m

def _solve(A: list[list[float]], b: list[float]) -> list[float]:
  "Solve A w = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [row[:] + [b[i]] for i, row in enumerate(A)]
  for c in range(n):
    piv = max(range(c, n), key=lambda r: abs(M[r][c]))
    M[c], M[piv] = M[piv], M[c]
    if abs(M[c][c]) < 1e-12: continue
    for r in range(n):
      if r != c and M[r][c]:
        f = M[r][c] / M[c][c]
        M[r] = [x - f * y for x, y in zip(M[r], M[c])]
  return [M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0
          for i in range(n)]

def _fit(X: list[list[float]], y: list[float]):
  "Ridge regression: centre X and y, then solve the normal equations."
  n, p = len(X), len(X[0])
  mx   = [sum(x[j] for x in X) / n for j in range(p)]
  my   = sum(y) / n
  Xc   = [[x[j] - mx[j] for j in range(p)] for x in X]
  yc   = [v - my for v in y]
  A    = [[sum(Xc[i][a] * Xc[i][b] for i in range(n)) + (RIDGE if a == b else 0.0)
           for b in range(p)] for a in range(p)]
  rhs  = [sum(Xc[i][a] * yc[i] for i in range(n)) for a in range(p)]
  return mx, my, _solve(A, rhs)

def _predict(model, x: list[float]) -> float:
  "The fitted linear model's guess for feature vector X."
  mx, my, w = model
  return my + sum(wj * (xj - mj) for wj, xj, mj in zip(w, x, mx))

def _linear(train: ezr.TBL, goal_at: int):
  "An off-the-shelf linear model for one goal: ridge regression on numeric inputs."
  feats = [c.at for c in train.x
           if any(_num(r[c.at]) for r in train.rows)
           and all(_num(r[c.at]) or r[c.at] == "?" for r in train.rows)]
  means = {at: _mean(train, at) for at in feats}
  X, y  = [], []
  for r in train.rows:
    if _num(r[goal_at]):
      X.append([_cell(r[at], means[at]) for at in feats])
      y.append(r[goal_at])
  if not feats or len(y) < 2:
    m = sum(y) / len(y) if y else 0.0
    return lambda r: m
  model = _fit(X, y)
  return lambda r: _predict(model, [_cell(r[at], means[at]) for at in feats])

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  """Lower is better for every entry.
  Tuners pick one row of TEST_CSV after at most BUDGET labels; the score is
  the ydist of the pick (0 is best). "ours" is step7.choose; "random" is
  random search with the same budget. The standard library has no tuner,
  so random search is its stand-in.
  Guessers predict every goal of each TEST_CSV row; the score is mean
  absolute error, scaled by each goal's range in TRAIN_CSV. "ours_guess" is
  step2.guess; "linreg_guess" is an off-the-shelf linear model fit on
  TRAIN_CSV."""
  train = ezr.Tbl(ezr.csv(train_csv))
  test  = ezr.Tbl(ezr.csv(test_csv))
  if train.names != test.names:
    raise ValueError("race: train and test need the same header")
  goals = [c for c in test.y if type(c) is ezr.Num]
  rows  = [r for r in test.rows if goals and all(_num(r[c.at]) for c in goals)]
  if not rows: raise ValueError("race: no test row has numeric goals")
  budget = min(BUDGET, len(rows))

  # tuners: each may label at most BUDGET rows of TEST_CSV
  def label(i: int) -> dict:
    return {test.names[c.at]: rows[i][c.at] for c in goals}

  path = _blind(test, rows)
  try:
    pick = step7.choose(path, label, budget=budget)
  finally:
    os.remove(path)
  ours = ezr.ydist(test, rows[pick])

  rng    = random.Random(ezr.the.Seed)
  sample = rng.sample(range(len(rows)), budget)
  rand   = min(ezr.ydist(test, rows[i]) for i in sample)

  # guessers: scale each goal's error by its range in TRAIN_CSV
  spans = {}
  for c in goals:
    vals = [r[c.at] for r in train.rows if _num(r[c.at])]
    span = (max(vals) - min(vals)) if vals else 1.0
    spans[test.names[c.at]] = span or 1.0

  models = {c.at: _linear(train, c.at) for c in goals}

  def ours_guess(r) -> dict:
    g = step2.guess(train_csv,
                    {test.names[c.at]: r[c.at] for c in test.x})
    return {test.names[c.at]: g.get(test.names[c.at], (0.0,))[0]
            for c in goals}

  def linreg_guess(r) -> dict:
    return {test.names[c.at]: models[c.at](r) for c in goals}

  def mae(guess_fn) -> float:
    total, n = 0.0, 0
    for r in rows:
      est = guess_fn(r)
      for c in goals:
        name   = test.names[c.at]
        total += abs(est[name] - r[c.at]) / spans[name]
        n     += 1
    return total / n

  return {
    "ours":         ours,
    "random":       rand,
    "ours_guess":   mae(ours_guess),
    "linreg_guess": mae(linreg_guess),
  }

