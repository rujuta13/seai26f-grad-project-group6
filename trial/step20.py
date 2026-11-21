# step20.py: race. Does our shortlist beat a top-OVR list? Does our penalty guesser beat an off-the-shelf model?
"step20.py: race. Does our shortlist beat a top-OVR list? Does our penalty guesser beat an off-the-shelf model?"
import math
import random
import ezr
from step2 import guess
from step7 import choose

BUDGET     = 30    # labels each randomised method may spend (step7's default)
REPEATS    = 5     # seeded runs of the randomised methods, averaged
RIDGE      = 1.0   # ridge penalty for the off-the-shelf regression
MAX_LEVELS = 12    # symbol columns with more levels than this are left out of the regression

def _is_num(v) -> bool:
  "True if V is a usable finite number (bools do not count)."
  return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _col_at(tbl, word: str) -> int:
  "Position of the column whose name is WORD, ignoring case."
  for at, name in enumerate(tbl.names):
    if str(name).strip().lower() == word: return at
  raise ValueError(f"no column named {word.upper()}")

def _penalty_col(tbl):
  "The penalty goal column, named like Penalties+ (any goal suffix)."
  for c in tbl.y:
    if c.txt.rstrip("+-!").strip().lower() == "penalties": return c
  raise ValueError(f"no penalty goal column; goals are {[c.txt for c in tbl.y]}")

def _labeller(tbl, rows):
  "A labeller that reveals a test row's goals, as a dict by goal name."
  def label(i):
    return {c.txt: rows[i][c.at] for c in tbl.y}
  return label

def _penalty_wins(test, pen):
  "Score a row 0..100 on the penalty goal alone: 100 at the best known penalty, 0 at the average."
  def gap(r):
    return abs(ezr.norm(pen, r[pen.at]) - pen.goal)
  gs  = [gap(r) for r in test.rows if _is_num(r[pen.at])]
  lo  = min(gs)
  avg = sum(gs) / len(gs)
  def score(r):
    if not _is_num(r[pen.at]): return -100.0
    return max(-100.0, min(100.0, 100 * (1 - (gap(r) - lo) / (avg - lo + 1e-32))))
  return score

def _top_overall(test) -> int:
  "Index of the row with the highest OVR: the simple 'top overall rating' list."
  at   = _col_at(test, "ovr")
  vals = [r[at] if _is_num(r[at]) else -math.inf for r in test.rows]
  return max(range(len(vals)), key=lambda i: vals[i])

def _guess_penalty(train_csv, train, test, pool) -> int:
  "Our penalty guesser: guess each candidate's penalty from its inputs; pick the one guessed closest to ideal."
  pen  = _penalty_col(train)
  best, best_d = pool[0], math.inf
  for i in pool:
    row  = test.rows[i]
    d    = {c.txt: row[c.at] for c in test.x}   # inputs only: goals stay unseen
    pred = guess(train_csv, d)[pen.txt][0]        # the median guess for the penalty goal
    gap  = abs(ezr.norm(pen, pred) - pen.goal)
    if gap < best_d: best, best_d = i, gap
  return best

def _solve(A, b):
  "Solve A x = b by Gauss-Jordan elimination with partial pivoting; singular columns give 0."
  n = len(b)
  M = [A[i][:] + [b[i]] for i in range(n)]
  for c in range(n):
    piv = max(range(c, n), key=lambda r: abs(M[r][c]))
    M[c], M[piv] = M[piv], M[c]
    p = M[c][c]
    if abs(p) < 1e-12: continue
    M[c] = [v / p for v in M[c]]
    for r in range(n):
      if r != c and M[r][c]:
        f = M[r][c]
        M[r] = [a - f * v for a, v in zip(M[r], M[c])]
  return [M[i][n] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]

def _ridge(X, y, lam):
  "Ridge least squares: beta minimising |X beta - y|^2 + lam |beta|^2, intercept (first term) unpenalised."
  p = len(X[0])
  A = [[0.0] * p for _ in range(p)]
  b = [0.0] * p
  for x, t in zip(X, y):
    for i in range(p):
      xi = x[i]
      if xi:
        b[i] += xi * t
        ai = A[i]
        for j in range(p): ai[j] += xi * x[j]
  for i in range(1, p): A[i][i] += lam
  return _solve(A, b)

def _ols_penalty(train, test, pool) -> int:
  "Off-the-shelf model: a linear regression of the penalty goal on the inputs; pick the candidate predicted closest to ideal."
  pen  = _penalty_col(train)
  spec = []                       # numeric inputs standardised; low-cardinality symbols one-hot
  for c in train.x:
    if type(c) is ezr.Sym:
      levels = sorted(c.has, key=str)
      if 0 < len(levels) <= MAX_LEVELS:
        spec += [("sym", c.at, lvl) for lvl in levels]
    else:
      spec.append(("num", c.at, c.mu, c.sd if c.sd > 0 else 1.0))

  def feats(row):
    out = [1.0]
    for s in spec:
      v = row[s[1]]
      if s[0] == "num":
        out.append(((v if _is_num(v) else s[2]) - s[2]) / s[3])
      else:
        out.append(1.0 if v == s[2] else 0.0)
    return out

  known = [r for r in train.rows if _is_num(r[pen.at])]
  if not known: raise ValueError("no training row has a known penalty")
  X    = [feats(r) for r in known]
  y    = [float(r[pen.at]) for r in known]
  beta = _ridge(X, y, RIDGE)

  best, best_d = pool[0], math.inf
  for i in pool:
    x    = feats(test.rows[i])
    pred = sum(b * v for b, v in zip(beta, x))
    gap  = abs(ezr.norm(pen, pred) - pen.goal)
    if gap < best_d: best, best_d = i, gap
  return best

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  """Each method's pick on TEST_CSV, scored by wins (100 = the best, 0 = an average row; higher is better).
  Keys over all goals: "ours" (step7 shortlist), "random" (random search with the same budget),
  "top_overall" (the row with the highest OVR).
  Keys on the penalty goal alone: "penalty_guesser" (step2 guess) and "penalty_ols" (off-the-shelf regression).
  Penalty picks are drawn from test rows whose penalty is known."""
  train = ezr.Tbl(ezr.csv(train_csv))
  test  = ezr.Tbl(ezr.csv(test_csv))
  rows  = test.rows
  if not rows: raise ValueError("no test rows")
  if not test.y: raise ValueError("no goal columns (names ending in + or -)")
  pen  = _penalty_col(test)
  pool = [i for i, r in enumerate(rows) if _is_num(r[pen.at])]
  if not pool: raise ValueError("no test row has a known penalty")

  wins_all = ezr.wins(test)
  budget   = min(BUDGET, len(rows))
  label    = _labeller(test, rows)
  seed0    = ezr.the.Seed
  runs     = {"ours": [], "random": []}
  try:
    for k in range(REPEATS):
      ezr.the.Seed = seed0 + k
      rng = random.Random(ezr.the.Seed)
      runs["ours"].append(wins_all(rows[choose(test_csv, label, budget)]))
      picks = rng.sample(range(len(rows)), budget)
      best  = min(picks, key=lambda i: ezr.ydist(test, rows[i]))
      runs["random"].append(wins_all(rows[best]))
  finally:
    ezr.the.Seed = seed0

  out = {name: sum(vs) / len(vs) for name, vs in runs.items()}
  out["top_overall"] = wins_all(rows[_top_overall(test)])
  wins_pen = _penalty_wins(test, pen)
  out["penalty_guesser"] = wins_pen(rows[_guess_penalty(train_csv, train, test, pool)])
  out["penalty_ols"]     = wins_pen(rows[_ols_penalty(train, test, pool)])
  return out

