# step10.py
import os, random, tempfile
import ezr
import step2
import step7

# The standard library ships no hyperparameter tuner and no regression
# model, so the rivals are built from stdlib pieces:
#   random : random search, the same label budget spent at random
#   grid   : a fixed, evenly spaced sweep of the candidates
#   linear : a least-squares linear model, trained on train_csv, that
#            picks the test row it predicts to be best (no test labels)
# "ours" is step7's tuner, run on a copy of test_csv with goal cells hidden.

def _best_of(tbl: ezr.TBL, idx: list[int]) -> int:
  "Among the labelled rows IDX, the index of the one with the smallest ydist."
  labelled = [tbl.rows[i] for i in idx]
  done     = ezr.clone(tbl, labelled)
  return min(idx, key=lambda i: ezr.ydist(done, tbl.rows[i]))

def _random(tbl: ezr.TBL, budget: int) -> int:
  "Random search: label BUDGET random rows, keep the best."
  rng = random.Random(ezr.the.Seed)
  idx = rng.sample(range(len(tbl.rows)), min(budget, len(tbl.rows)))
  return _best_of(tbl, idx)

def _grid(tbl: ezr.TBL, budget: int) -> int:
  "Grid search: label BUDGET evenly spaced rows, keep the best."
  n   = len(tbl.rows)
  k   = min(budget, n)
  idx = sorted({int(i * n / k) for i in range(k)})
  return _best_of(tbl, idx)

def _ours(test_csv: str, budget: int) -> int:
  "Our tuner (step7), blind: goal cells are hidden in a temp copy and labelled on demand."
  src     = list(ezr.csv(test_csv))
  headers = [str(h) for h in src[0]]
  goal    = [h[-1] in "+-" for h in headers]

  def label(i: int) -> dict:          # true goal values of data row I
    row = src[i + 1]
    return {h: v for h, v, g in zip(headers, row, goal) if g}

  fd, path = tempfile.mkstemp(suffix=".csv")
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(",".join(headers) + "\n")
      for row in src[1:]:
        f.write(",".join("?" if g else str(v)
                         for v, g in zip(row, goal)) + "\n")
    return step7.choose(path, label, budget=budget)
  finally:
    os.remove(path)

def _feats(tbl: ezr.TBL, row: ezr.ROW) -> list[float]:
  "A bias term, then one feature per x column: norm for numbers, one-hot for symbols."
  out = [1.0]
  for c in tbl.x:
    v = row[c.at]
    if type(c) is ezr.Sym:
      out += [1.0 if v == k else 0.0 for k in c.has]
    else:
      out.append(0.5 if v == "?" else ezr.norm(c, v))
  return out

def _solve(A: list[list[float]], b: list[float]) -> list[float]:
  "Solve A w = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [A[i][:] + [b[i]] for i in range(n)]
  for i in range(n):
    p = max(range(i, n), key=lambda r: abs(M[r][i]))
    M[i], M[p] = M[p], M[i]
    piv = M[i][i] or 1e-32
    for r in range(n):
      if r != i:
        f = M[r][i] / piv
        for k in range(i, n + 1): M[r][k] -= f * M[i][k]
  return [M[i][n] / (M[i][i] or 1e-32) for i in range(n)]

def _fit(X: list[list[float]], y: list[float], lam: float = 1e-6) -> list[float]:
  "Ridge least squares via the normal equations."
  d = len(X[0])
  A = [[sum(x[i] * x[j] for x in X) + (lam if i == j else 0.0)
        for j in range(d)] for i in range(d)]
  b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(d)]
  return _solve(A, b)

def _linear(tr: ezr.TBL, test: ezr.TBL) -> int:
  "A linear model of the train wins; pick the test row it predicts is best."
  score = ezr.wins(tr)
  X     = [_feats(tr, r) for r in tr.rows]
  y     = [score(r) for r in tr.rows]
  w     = _fit(X, y)
  preds = [sum(a * b for a, b in zip(w, _feats(tr, r))) for r in test.rows]
  return max(range(len(preds)), key=lambda i: preds[i])

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Each method picks one row of TEST_CSV; score it by wins on test (100 = best, 0 = average)."
  test = ezr.Tbl(ezr.csv(test_csv))
  if not test.rows: raise ValueError(f"no rows in {test_csv}")
  budget = min(ezr.the.Stop, len(test.rows))
  picks  = {
    "ours":   _ours(test_csv, budget),
    "random": _random(test, budget),
    "grid":   _grid(test, budget),
    "linear": _linear(step2._table(train_csv), test),
  }
  score = ezr.wins(test)
  return {k: float(score(test.rows[i])) for k, i in picks.items()}

