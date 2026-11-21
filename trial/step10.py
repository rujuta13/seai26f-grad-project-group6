# step10.py
#!/usr/bin/env python3 -B
"""
step10.py: race.  Our tuner and guesser against standard rivals on test_csv.

The standard library ships no tuner or regression model, so the rival is a
stdlib-only ridge regression (solved by Gauss-Jordan elimination) that guides
a shortlist search.  Random search is the second rival.

Keys in the returned dict, lower is better for every key:
  ours        step7.choose (ezr.acquire under a label budget); score is the
              true ydist of the chosen build, where 0 is ideal.
  random      random search with the same budget and the same pick rule.
  ols         the ridge model ranks test builds by predicted ydist; its top
              BUDGET builds are labelled and the best of them is chosen.
  guess_ours  step2.guess; score is the mean |guess - truth| per goal,
              in training standard-deviation units.
  guess_ols   the ridge model's predicted goals, scored the same way.
Reuses step2 (guess) and step7 (choose) for ours; ezr for tables and distance.
"""
import os, random, tempfile
import ezr
import step2, step7

def _feats(tbl) -> list:
  "Design columns for the inputs: a number as-is, a symbol as one indicator per value."
  out = []
  for c in tbl.x:
    if type(c) is ezr.Num:
      out.append((c.at, None))
    else:
      vs = sorted({r[c.at] for r in tbl.rows if r[c.at] != "?"}, key=str)
      out += [(c.at, v) for v in vs]
  return out

def _encode(feats, row) -> list:
  "One row as numbers: an intercept, then each design column (missing = 0)."
  out = [1.0]
  for at, v in feats:
    x = row[at]
    if v is None:
      out.append(float(x) if x != "?" else 0.0)
    else:
      out.append(1.0 if x == v else 0.0)
  return out

def _solve(A: list, b: list) -> list:
  "Solve A x = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [A[i][:] + [b[i]] for i in range(n)]
  for i in range(n):
    p = max(range(i, n), key=lambda r: abs(M[r][i]))
    M[i], M[p] = M[p], M[i]
    for r in range(n):
      if r != i and M[r][i] != 0:
        f = M[r][i] / M[i][i]
        M[r] = [a - f*c for a, c in zip(M[r], M[i])]
  return [M[i][n] / M[i][i] for i in range(n)]

def _ols(train) -> tuple:
  "Ridge regression of each goal on the inputs, fitted on the fully labelled training rows."
  feats = _feats(train)
  done  = [r for r in train.rows if all(r[c.at] != "?" for c in train.y)]
  X     = [_encode(feats, r) for r in done]
  k     = len(feats) + 1
  A     = [[0.0]*k for _ in range(k)]
  for x in X:
    for i in range(k):
      for j in range(k):
        A[i][j] += x[i]*x[j]
  for i in range(k):
    A[i][i] += 1e-2          # ridge: one-hot columns are collinear
  betas = {}
  for c in train.y:
    b = [0.0]*k
    for x, r in zip(X, done):
      for i in range(k):
        b[i] += x[i]*r[c.at]
    betas[c.txt] = _solve(A, b)
  return feats, betas

def _predict(feats, betas, row) -> dict:
  "Predicted goal values for one ROW, keyed by goal header name."
  x = _encode(feats, row)
  return {g: sum(bi*xi for bi, xi in zip(b, x)) for g, b in betas.items()}

def _write_blind(tbl, goal_at, path: str) -> None:
  "Write TBL to PATH with every goal cell set to '?'."
  with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(tbl.names) + "\n")
    for r in tbl.rows:
      f.write(",".join("?" if at in goal_at else str(v)
                       for at, v in enumerate(r)) + "\n")

def _guess_error(train, rows, guesses) -> float:
  "Mean |guess - truth| per known goal, in training sd units."
  errs = [abs(est[c.txt] - row[c.at]) / (c.sd or 1)
          for row, est in zip(rows, guesses)
          for c in train.y if row[c.at] != "?"]
  return sum(errs)/len(errs) if errs else 0.0

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Tuner and guesser scores on test_csv, one entry per method; lower is better."
  random.seed(ezr.the.Seed)
  train   = ezr.Tbl(ezr.csv(train_csv))
  full    = ezr.Tbl(ezr.csv(test_csv))
  n       = len(full.rows)
  if n == 0: raise ValueError("no rows in test_csv")
  names   = full.names
  goal_at = [c.at for c in full.y]
  goal_nm = {names[at] for at in goal_at}
  budget  = min(ezr.the.Stop, n)

  def true(i) -> float:
    "The real score of test row I (all goals known here): lower is better."
    return ezr.ydist(full, full.rows[i])

  def label(i) -> dict:
    "Reveal test row I's goals, in the form step7's oracle expects."
    return {names[at]: full.rows[i][at] for at in goal_at}

  def pick(idx) -> int:
    "The best of the labelled rows IDX, judged on those labels alone (as step7 does)."
    scorer = ezr.clone(full, [full.rows[i] for i in idx])
    return min(idx, key=lambda i: ezr.ydist(scorer, full.rows[i]))

  out = {}

  with tempfile.TemporaryDirectory() as d:
    blind = os.path.join(d, "blind.csv")
    _write_blind(full, goal_at, blind)
    out["ours"] = true(step7.choose(blind, label, budget))

  out["random"] = true(pick(random.sample(range(n), budget)))

  feats, betas = _ols(train)
  guessed = [_predict(feats, betas, r) for r in full.rows]

  def predicted(i) -> float:
    "Predicted ydist of test row I, using the model's goals and the training table's scale."
    probe = list(full.rows[i])
    for c in train.y: probe[c.at] = guessed[i][c.txt]
    return ezr.ydist(train, probe)

  shortlist = sorted(range(n), key=predicted)[:budget]
  out["ols"] = true(pick(shortlist))

  inputs = [{names[at]: v for at, v in enumerate(r) if names[at] not in goal_nm}
            for r in full.rows]
  ours_g = [{g: est[0] for g, est in step2.guess(train_csv, inp).items()}
            for inp in inputs]
  out["guess_ours"] = _guess_error(train, full.rows, ours_g)
  out["guess_ols"]  = _guess_error(train, full.rows, guessed)
  return out

