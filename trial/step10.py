# step10.py
"""Step 10: race.  Race our tuner against random search and a stdlib-only
local search, and our guesser against an off-the-shelf linear model.

Every score is lower-is-better.  Tuner scores are the ydist (distance from
the ideal, 0 is best) of the build each method picks from TEST_CSV.  Guesser
scores are the mean normalised goal error over test rows (0 is best).  The
keys are: "ours" (our tuner, step7), "random_search", "local_search",
"guess_ours" (our guesser, step2) and "linear_ridge".

The client's "build speed" is not a column of its own, so the guesser score
averages over every goal column.

Uses only the standard library plus ezr, and our earlier steps 2 and 7.
The standard library ships no tuner or regressor, so the rivals are:
  random_search  label BUDGET random builds, keep the best.
  local_search   start at a random build, then repeatedly label the
                 unlabelled build nearest the best one so far.
  linear_ridge   least-squares linear regression with a small ridge
                 penalty, solved in pure Python.
"""
import math
import os
import random
import statistics
import tempfile
import ezr
import step2
import step7

BUDGET = 30     # labels each tuner may spend, as in step7
RIDGE  = 1e-3   # ridge penalty for the linear rival
KEYS   = ("ours", "random_search", "local_search",
          "guess_ours", "linear_ridge")


def _load(path: str) -> ezr.TBL:
  "Read PATH into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(path))


def _mean(xs: list[float]) -> float:
  "Mean of XS, or infinity if there is nothing to score."
  return sum(xs) / len(xs) if xs else math.inf


def _blind(tbl: ezr.TBL, path: str) -> None:
  "Write TBL to PATH with every goal cell '?', so the tuner cannot peek."
  goal_at = {c.at for c in tbl.y}
  with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(str(n) for n in tbl.names) + "\n")
    for row in tbl.rows:
      cells = ["?" if at in goal_at else str(v)
               for at, v in enumerate(row)]
      f.write(",".join(cells) + "\n")


def _near(tbl: ezr.TBL, a: int, b: int) -> float:
  "Distance between rows A and B of TBL over the x columns (0 if none)."
  return ezr.xdist(tbl, tbl.rows[a], tbl.rows[b]) if tbl.x else 0.0


def _local(tbl: ezr.TBL, ys: list[float], budget: int,
           rng: random.Random) -> float:
  "Local search: label the nearest unlabelled build to the best so far."
  n    = len(tbl.rows)
  best = rng.randrange(n)
  done = {best}
  while len(done) < budget:
    nxt = min((z for z in range(n) if z not in done),
              key=lambda z: (_near(tbl, best, z), z))
    done.add(nxt)
    if ys[nxt] < ys[best]: best = nxt
  return min(ys[i] for i in done)


def _guess_error(train_csv: str, train: ezr.TBL,
                 test: ezr.TBL) -> float:
  "Mean normalised goal error of step2's guesses over the test rows."
  errs = []
  for r in test.rows:
    d    = dict(zip(test.names, r))
    flags = {c.txt: d.get(c.txt, "?") for c in train.x}
    est  = step2.guess(train_csv, flags)
    for c in train.y:
      if not isinstance(c, ezr.Num) or c.txt not in est: continue
      actual = d.get(c.txt, "?")
      if not isinstance(actual, (int, float)): continue
      errs.append(abs(ezr.norm(c, est[c.txt][0])
                      - ezr.norm(c, actual)))
  return _mean(errs)


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
  "Solve A w = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [A[i][:] + [b[i]] for i in range(n)]
  for col in range(n):
    piv = max(range(col, n), key=lambda r: abs(M[r][col]))
    M[col], M[piv] = M[piv], M[col]
    if abs(M[col][col]) < 1e-12: continue
    for r in range(n):
      if r != col and M[r][col] != 0.0:
        f = M[r][col] / M[col][col]
        M[r] = [a - f * c for a, c in zip(M[r], M[col])]
  return [M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0
          for i in range(n)]


def _ridge(X: list[list[float]], y: list[float]) -> list[float]:
  "Ridge least squares: weights w minimising |Xw - y|^2 + RIDGE |w|^2."
  p = len(X[0])
  A = [[0.0] * p for _ in range(p)]
  b = [0.0] * p
  for x, t in zip(X, y):
    for i in range(p):
      b[i] += x[i] * t
      xi, Ai = x[i], A[i]
      for j in range(p):
        Ai[j] += xi * x[j]
  for i in range(p): A[i][i] += RIDGE
  return _solve(A, b)


def _linear(train: ezr.TBL, test: ezr.TBL) -> float:
  "Off-the-shelf rival: linear regression on the flags, one model per goal."
  xs     = list(train.x)
  levels = {c.txt: sorted({r[c.at] for r in train.rows if r[c.at] != "?"},
                          key=str)
            for c in xs if not isinstance(c, ezr.Num)}
  means  = {}
  for c in xs:
    if isinstance(c, ezr.Num):
      known = [r[c.at] for r in train.rows
               if isinstance(r[c.at], (int, float))]
      means[c.txt] = sum(known) / len(known) if known else 0.0

  def feats(d: dict) -> list[float]:
    "Numbers for one row: a bias, numeric cells, and one-hot symbols."
    out = [1.0]
    for c in xs:
      v = d.get(c.txt, "?")
      if isinstance(c, ezr.Num):
        out.append(float(v) if isinstance(v, (int, float))
                   else means[c.txt])
      else:
        out += [1.0 if v == s else 0.0 for s in levels[c.txt]]
    return out

  X      = [feats(dict(zip(train.names, r))) for r in train.rows]
  models = {}
  for c in train.y:
    if not isinstance(c, ezr.Num): continue
    pairs = [(x, r[c.at]) for x, r in zip(X, train.rows)
             if isinstance(r[c.at], (int, float))]
    if not pairs: continue
    gs = [g for _, g in pairs]
    mu = sum(gs) / len(gs)
    sd = statistics.pstdev(gs) or 1.0
    w  = _ridge([x for x, _ in pairs], [(g - mu) / sd for g in gs])
    models[c.txt] = (w, mu, sd)

  errs = []
  for r in test.rows:
    d = dict(zip(test.names, r))
    x = feats(d)
    for c in train.y:
      if c.txt not in models: continue
      actual = d.get(c.txt, "?")
      if not isinstance(actual, (int, float)): continue
      w, mu, sd = models[c.txt]
      pred = mu + sd * sum(a * b for a, b in zip(w, x))
      errs.append(abs(ezr.norm(c, pred) - ezr.norm(c, actual)))
  return _mean(errs)


def race(train_csv: str, test_csv: str) -> dict[str, float]:
  """Score each method on TEST_CSV, lower is better.  The tuners each spend
  at most BUDGET labels picking a build; "ours" is step7.choose.  The
  guessers are scored on how close their goal guesses land to the truth.
  TRAIN_CSV is used only by the guessers.  Always includes the key "ours"."""
  train = _load(train_csv)
  test  = _load(test_csv)
  if not test.rows or not test.y or not train.y:
    return {k: math.inf for k in KEYS}
  rng    = random.Random(ezr.the.Seed)
  n      = len(test.rows)
  budget = min(BUDGET, n)
  ys     = [ezr.ydist(test, r) for r in test.rows]
  out    = {}

  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "blind.csv")
    _blind(test, path)

    def label(i: int) -> dict:
      "Reveal the goals of test build I, as an oracle would."
      return {c.txt: test.rows[i][c.at] for c in test.y}

    out["ours"] = ys[step7.choose(path, label, budget=budget)]

  out["random_search"] = min(ys[i] for i in rng.sample(range(n), budget))
  out["local_search"]  = _local(test, ys, budget, rng)
  out["guess_ours"]    = _guess_error(train_csv, train, test)
  out["linear_ridge"]  = _linear(train, test)
  return out

