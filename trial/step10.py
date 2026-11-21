# step10.py
"Step 10: race. Our tuner and guesser against rivals on TEST_CSV, all scores lower-is-better."
# The standard library ships no tuner, and third-party models are not allowed,
# so the rivals are: random search (tuner), a stdlib hill-climber (tuner stand-in),
# and ridge regression written with the stdlib (off-the-shelf model stand-in).
# Tuner scores are the goal distance of the chosen build (0 is ideal).
# Guesser scores are the mean goal error, in train standard deviations.
# Compare scores within each group, not across groups.
import math
import random
import ezr
import step2
import step7

BUDGET = 30          # labels each tuner may spend (same as step7's default)
RIDGE  = 1.0         # penalty on the linear model's weights
SEED   = 1234567891  # fixed seed, so results repeat

def _load(csv_path: str):
  "Table of CSV_PATH, with columns and goals from its header."
  return ezr.Tbl(ezr.csv(csv_path))

def _numgoals(tbl) -> list:
  "The numeric goal columns of TBL."
  return [c for c in tbl.y if type(c) is ezr.Num]

def _sd(c) -> float:
  "A column's standard deviation, or 1.0 if it has none."
  return c.sd if c.sd > 0 else 1.0

def _mean(xs: list) -> float:
  "Mean of XS, or infinity if XS is empty."
  return sum(xs) / len(xs) if xs else math.inf

def _dist(gtr, vals: dict) -> float:
  "Distance of goal values VALS (name -> value) from the ideal, scaled by train goals GTR."
  sq, n = 0.0, 0
  for c in gtr:
    v = vals.get(c.txt, "?")
    if not isinstance(v, (int, float)): continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else math.inf

def _solve(A: list, B: list) -> list:
  "Solve A X = B by Gauss-Jordan elimination with partial pivoting."
  n = len(A)
  M = [A[i][:] + B[i][:] for i in range(n)]
  for c in range(n):
    piv = max(range(c, n), key=lambda r: abs(M[r][c]))
    M[c], M[piv] = M[piv], M[c]
    p = M[c][c]
    M[c] = [x / p for x in M[c]]
    for r in range(n):
      if r != c and M[r][c] != 0.0:
        f = M[r][c]
        M[r] = [a - f * b for a, b in zip(M[r], M[c])]
  return [row[n:] for row in M]

def _tuner_rivals(test_csv: str, gtr) -> dict[str, float]:
  "Score ours (step7.choose), random search, and a hill-climber on the same blind builds."
  tbl  = _load(test_csv)
  rows = tbl.rows
  n    = len(rows)
  if n == 0: raise ValueError("race: no builds in test_csv")
  budget = min(BUDGET, n)

  def label(i: int) -> dict:
    "The true goal values of build I: the oracle, billed once per call."
    return {c.txt: rows[i][c.at] for c in tbl.y}

  ours_i = step7.choose(test_csv, label, budget=budget)
  ours   = _dist(gtr, label(ours_i))

  picks  = random.Random(SEED).sample(range(n), budget)
  rand   = min(_dist(gtr, label(i)) for i in picks)

  rng    = random.Random(SEED)
  cur    = rng.randrange(n)
  seen   = {cur: _dist(gtr, label(cur))}
  while len(seen) < budget:
    best = min(seen, key=lambda i: (seen[i], i))
    free = [z for z in range(n) if z not in seen]
    nxt  = min(free, key=lambda z: (ezr.xdist(tbl, rows[best], rows[z]), z))
    seen[nxt] = _dist(gtr, label(nxt))
  hill   = min(seen.values())

  return {"ours": ours, "random": rand, "hillclimb": hill}

def _guess_rivals(train_csv: str, test_csv: str) -> dict[str, float]:
  "Score ours (step2.guess) and ridge regression on the same test builds."
  ttr = _load(train_csv)
  gtr = _numgoals(ttr)
  tst = _load(test_csv)

  # ours: step2's neighbour median, from input columns only
  errs = []
  for row in tst.rows:
    inputs = {c.txt: row[c.at] for c in tst.x}
    est    = step2.guess(train_csv, inputs)
    for c in gtr:
      truth = row[c.at]
      if not isinstance(truth, (int, float)) or c.txt not in est: continue
      errs.append(abs(est[c.txt][0] - truth) / _sd(c))
  guess_ours = _mean(errs)

  # linear: ridge regression on standardised numbers and one-hot symbols
  fit = [r for r in ttr.rows
         if all(isinstance(r[c.at], (int, float)) for c in gtr)]
  if not fit: raise ValueError("race: no train build has every goal")
  levels = {c.at: sorted(c.has, key=str) for c in ttr.x
            if type(c) is ezr.Sym}

  def feats(row: list) -> list:
    "Full-width ROW into a feature vector, with a leading intercept."
    f = [1.0]
    for c in ttr.x:
      v = row[c.at]
      if type(c) is ezr.Num:
        f.append(0.0 if v == "?" else (v - c.mu) / _sd(c))
      else:
        f.extend(1.0 if v == s else 0.0 for s in levels[c.at])
    return f

  X = [feats(r) for r in fit]
  Y = [[(r[c.at] - c.mu) / _sd(c) for c in gtr] for r in fit]
  p = len(X[0])
  A = [[0.0] * p for _ in range(p)]
  B = [[0.0] * len(gtr) for _ in range(p)]
  for x, y in zip(X, Y):
    for i in range(p):
      for j in range(p): A[i][j] += x[i] * x[j]
      for k in range(len(gtr)): B[i][k] += x[i] * y[k]
  for i in range(p): A[i][i] += RIDGE
  W = _solve(A, B)

  errs = []
  for row in tst.rows:
    f = feats(row)
    for k, c in enumerate(gtr):
      truth = row[c.at]
      if not isinstance(truth, (int, float)): continue
      pred = sum(fi * W[i][k] for i, fi in enumerate(f))
      errs.append(abs(pred - (truth - c.mu) / _sd(c)))
  linear = _mean(errs)

  return {"guess_ours": guess_ours, "linear": linear}

def race(train_csv, test_csv) -> dict[str, float]:
  """Race our tuner (step7) and guesser (step2) against rivals on test_csv.
  Returns {method: score}, including the key "ours" (our tuner).
  Tuner keys: ours, random, hillclimb (goal distance of the chosen build).
  Guesser keys: guess_ours, linear (mean goal error, in train sd).
  All scores are lower-is-better."""
  gtr = _numgoals(_load(train_csv))
  if not gtr: raise ValueError("race: no numeric goal to race on")
  out = _tuner_rivals(test_csv, gtr)
  out.update(_guess_rivals(train_csv, test_csv))
  return {k: float(v) for k, v in out.items()}

