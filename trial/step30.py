# step30.py
import os
import random
import tempfile
from collections.abc import Callable
import ezr
import step17
import step22

BUDGET = 30           # labels each targeting method may spend on test rows
RIDGE  = 1e-6         # a tiny ridge keeps the off-the-shelf linear fit stable
RESP   = "response"   # the goal the response guesser is judged on
INC    = "income"     # the input the highest-income rival is ranked by

def _num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _find(tbl: ezr.TBL, base: str) -> int:
  "Column index of the column whose name, sign dropped and in lower case, is BASE."
  for at, s in enumerate(tbl.names):
    if s.rstrip("+-").lower() == base: return at
  raise ValueError(f"race: no column named {base!r}")

def _rank(v) -> float:
  "Sort key for INC: the highest income first; unknown incomes last."
  return -v if _num(v) else float("inf")

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

def _linear(train: ezr.TBL, goal_at: int) -> Callable[[ezr.ROW], float]:
  "An off-the-shelf linear model for one goal: ridge regression on the numeric inputs."
  feats = [c.at for c in train.x if any(_num(r[c.at]) for r in train.rows)]
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
  """Lower is better for every entry, and every entry is measured on TEST_CSV.
  Targeting question: each method may label at most BUDGET test rows and must
  return one row to mail; the score is the ydist of its pick (0 is best).
  "ours" is step17.choose. "top_income" labels the BUDGET highest-income rows
  and keeps the best. "random" labels BUDGET random rows and keeps the best.
  Response question: each guesser predicts RESPONSE for every test row that
  has one; the score is mean absolute error, scaled by RESPONSE's range in
  TRAIN_CSV. "ours_guess" is step22.guess; "linreg_guess" is an off-the-shelf
  linear model fit on TRAIN_CSV.
  The two kinds of score are on different scales: compare within a kind."""
  train = ezr.Tbl(ezr.csv(train_csv))
  test  = ezr.Tbl(ezr.csv(test_csv))
  if train.names != test.names:
    raise ValueError("race: train and test need the same header")
  goals = [c for c in test.y if type(c) is ezr.Num]
  if not goals: raise ValueError("race: the test file has no numeric goals")
  rows  = [r for r in test.rows if all(_num(r[c.at]) for c in goals)]
  if not rows: raise ValueError("race: no test row has numeric goals")
  budget = min(BUDGET, len(rows))

  # targeting: each method labels at most BUDGET test rows
  def label(i: int) -> dict:
    return {test.names[c.at]: rows[i][c.at] for c in goals}

  path = _blind(test, rows)
  try:
    pick = step17.choose(path, label, budget=budget)
  finally:
    os.remove(path)
  ours = ezr.ydist(test, rows[pick])

  inc_at     = _find(test, INC)
  by_inc     = sorted(range(len(rows)), key=lambda i: _rank(rows[i][inc_at]))
  top_income = min(ezr.ydist(test, rows[i]) for i in by_inc[:budget])

  rng    = random.Random(ezr.the.Seed)
  sample = rng.sample(range(len(rows)), budget)
  rand   = min(ezr.ydist(test, rows[i]) for i in sample)

  # response guessers: error scaled by the goal's range in TRAIN_CSV
  resp_at   = _find(test, RESP)
  resp_name = test.names[resp_at]
  vals      = [r[resp_at] for r in train.rows if _num(r[resp_at])]
  if not vals: raise ValueError("race: no numeric response in TRAIN_CSV")
  span      = (max(vals) - min(vals)) or 1.0
  resp_rows = [r for r in test.rows if _num(r[resp_at])]
  if not resp_rows: raise ValueError("race: no test row has a response value")

  model = _linear(train, resp_at)

  def ours_guess(r: ezr.ROW) -> float:
    cells = {test.names[c.at]: r[c.at] for c in test.x}
    g     = step22.guess(train_csv, cells)
    return g.get(resp_name, (0.0, 0.0, 0.0))[0]

  def linreg_guess(r: ezr.ROW) -> float:
    return model(r)

  def mae(guess: Callable[[ezr.ROW], float]) -> float:
    total = sum(abs(guess(r) - r[resp_at]) / span for r in resp_rows)
    return total / len(resp_rows)

  return {
    "ours":         ours,
    "top_income":   top_income,
    "random":       rand,
    "ours_guess":   mae(ours_guess),
    "linreg_guess": mae(linreg_guess),
  }

