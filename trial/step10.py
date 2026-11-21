# step10.py: race. Our tuner and guesser against standard-library rivals, on the same test rows.
"step10.py: race. Our tuner and guesser against standard-library rivals, on the same test rows."
# The standard library has no tuner or regression model, so the rivals are
# stdlib-only stand-ins: random search, a local-search tuner (hill_climb),
# and ordinary least squares (ols) fitted with a small ridge term.
import math
import random
import ezr
from step2 import guess
from step7 import choose

BUDGET  = 30   # labels each tuner may spend (step7's default)
REPEATS = 5    # seeded runs of the randomised tuners, averaged

def _labeller(tbl, rows):
  "A labeller that reveals a test row's goals, as a dict by goal name."
  def label(i):
    return {c.txt: rows[i][c.at] for c in tbl.y}
  return label

def _random(tbl, rows, rng, budget):
  "Random search: label BUDGET random rows, keep the one closest to ideal."
  picks = rng.sample(range(len(rows)), budget)
  return min(picks, key=lambda i: ezr.ydist(tbl, rows[i]))

def _hill(tbl, rows, rng, budget):
  "Local search: label a random start, then repeatedly label the unlabelled row nearest the best so far."
  todo   = rng.sample(range(len(rows)), len(rows))
  best   = todo.pop()
  best_y = ezr.ydist(tbl, rows[best])
  used   = 1
  while todo and used < budget:
    todo.sort(key=lambda i: ezr.xdist(tbl, rows[i], rows[best]))
    i = todo.pop(0)
    used += 1
    y = ezr.ydist(tbl, rows[i])
    if y < best_y: best, best_y = i, y
  return best

def _solve(A, b):
  "Solve A x = b by Gauss-Jordan elimination with partial pivoting; singular columns give 0."
  n = len(b)
  M = [A[i][:] + [b[i]] for i in range(n)]
  for col in range(n):
    piv = max(range(col, n), key=lambda r: abs(M[r][col]))
    M[col], M[piv] = M[piv], M[col]
    p = M[col][col]
    if abs(p) < 1e-12: continue
    for r in range(n):
      if r != col and M[r][col]:
        f = M[r][col] / p
        M[r] = [a - f * c for a, c in zip(M[r], M[col])]
  return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
          for i in range(n)]

def _fit(X, y, lam=1e-6):
  "Ridge-regularised least squares: the coefficients beta minimising |X beta - y|."
  p = len(X[0])
  A = [[sum(x[i] * x[j] for x in X) + (lam if i == j else 0.0)
        for j in range(p)] for i in range(p)]
  b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(p)]
  return _solve(A, b)

def _ols_pick(train_csv, test) -> int:
  "Off-the-shelf model: one linear regression per goal on the inputs; pick the test row predicted closest to ideal."
  train = ezr.Tbl(ezr.csv(train_csv))
  spec, mus = [], {}          # one feature per numeric input, one per symbol level
  for c in train.x:
    if type(c) is ezr.Sym:
      spec += [(c.at, v) for v in sorted(c.has, key=str)]
    else:
      spec.append((c.at, None))
      mus[c.at] = c.mu

  def feats(row):
    out = [1.0]
    for at, lvl in spec:
      v = row[at]
      if lvl is None:
        out.append(float(v) if isinstance(v, (int, float)) else mus[at])
      else:
        out.append(1.0 if v == lvl else 0.0)
    return out

  full = [r for r in train.rows if all(r[c.at] != "?" for c in train.y)]
  if not full: raise ValueError("no training row has all its goals known")
  X     = [feats(r) for r in full]
  betas = {c.at: _fit(X, [float(r[c.at]) for r in full]) for c in train.y}

  best, best_d = 0, math.inf
  for i, row in enumerate(test.rows):
    x    = feats(row)
    gaps = []
    for c in train.y:
      pred = sum(b * v for b, v in zip(betas[c.at], x))
      gaps.append(abs(ezr.norm(c, pred) - c.goal))
    d = ezr.dist(gaps, len(gaps))
    if d < best_d: best, best_d = i, d
  return best

def _guess_pick(train_csv, test) -> int:
  "Our guesser: for each test row, guess its goals from its inputs alone; pick the row guessed closest to ideal."
  train = ezr.Tbl(ezr.csv(train_csv))
  best, best_d = 0, math.inf
  for i, row in enumerate(test.rows):
    d    = {c.txt: row[c.at] for c in test.x}   # inputs only: goals are unknown
    pred = guess(train_csv, d)
    gaps = [abs(ezr.norm(g, pred[g.txt][0]) - g.goal) for g in train.y]
    dist = ezr.dist(gaps, len(gaps))
    if dist < best_d: best, best_d = i, dist
  return best

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  """Each method's pick on TEST_CSV, scored by wins (100 = the best test row,
  0 = an average one). Higher is better. Keys: ours (step7 tuner), random,
  hill_climb, guesser (step2), ols."""
  test = ezr.Tbl(ezr.csv(test_csv))
  rows = test.rows
  if not rows: raise ValueError("no test rows")
  if not test.y: raise ValueError("no goal columns (names ending in + or -)")
  score  = ezr.wins(test)
  label  = _labeller(test, rows)
  budget = min(BUDGET, len(rows))
  seed0  = ezr.the.Seed
  runs   = {"ours": [], "random": [], "hill_climb": []}
  try:
    for k in range(REPEATS):
      ezr.the.Seed = seed0 + k
      rng = random.Random(ezr.the.Seed)
      runs["ours"].append(score(rows[choose(test_csv, label, budget)]))
      runs["random"].append(score(rows[_random(test, rows, rng, budget)]))
      runs["hill_climb"].append(score(rows[_hill(test, rows, rng, budget)]))
  finally:
    ezr.the.Seed = seed0
  out = {name: sum(vs) / len(vs) for name, vs in runs.items()}
  out["guesser"] = score(rows[_guess_pick(train_csv, test)])
  out["ols"]     = score(rows[_ols_pick(train_csv, test)])
  return out

