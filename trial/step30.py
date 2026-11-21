# step30.py: race. Does our targeting beat mailing the highest-income customers?
# Does our response guesser beat an off-the-shelf model?
"step30.py: race. Does our targeting beat mailing the highest-income customers? Does our response guesser beat an off-the-shelf model?"
# Each method names one customer. Scores are on RESPONSE alone:
# 100 = the best known responder, 0 = an average customer.
import math
import os
import random
import tempfile
import ezr
from step22 import guess
from step27 import choose

BUDGET     = 30    # labels each randomised method may spend (step27's default)
REPEATS    = 5     # seeded runs of the randomised methods, averaged
RIDGE      = 1.0   # ridge penalty for the off-the-shelf regression
MAX_LEVELS = 12    # symbol columns with more levels than this are left out of the regression

def _is_num(v) -> bool:
  "True if V is a usable finite number (bools do not count)."
  return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _num_or_low(v) -> float:
  "V if it is a usable number, else minus infinity (so unknowns never win a max)."
  return v if _is_num(v) else -math.inf

def _col_at(tbl, word: str) -> int:
  "Position of the column whose name is WORD, ignoring case."
  for at, name in enumerate(tbl.names):
    if str(name).strip().lower() == word: return at
  raise ValueError(f"no column named {word.upper()}")

def _response_col(tbl):
  "The response goal column, named like RESPONSE+ (any goal suffix)."
  for c in tbl.y:
    if c.txt.rstrip("+-!").strip().lower() == "response": return c
  raise ValueError(f"no response goal column; goals are {[c.txt for c in tbl.y]}")

def _response_wins(E, resp):
  "Score a row 0..100 on the response goal alone: 100 at the best known responder, 0 at the average."
  def gap(r):
    return abs(ezr.norm(resp, r[resp.at]) - resp.goal)
  gs  = [gap(r) for r in E]
  lo  = min(gs)
  avg = sum(gs) / len(gs)
  def score(r):
    if not _is_num(r[resp.at]): return -100.0
    return max(-100.0, min(100.0, 100 * (1 - (gap(r) - lo) / (avg - lo + 1e-32))))
  return score

def _write_blind(names, rows, goal_ats, path) -> None:
  "Write ROWS to PATH with every goal cell set to '?', so the tuner sees inputs only."
  with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(str(n) for n in names) + "\n")
    for row in rows:
      cells = ["?" if at in goal_ats else str(v) for at, v in enumerate(row)]
      f.write(",".join(cells) + "\n")

def _top_income(E, at) -> int:
  "Index of the customer with the highest income: the 'mail the richest' rule."
  return max(range(len(E)), key=lambda i: _num_or_low(E[i][at]))

def _guess_pick(train_csv, test, E, resp) -> int:
  "Our response guesser: guess each customer's response from inputs alone; pick the highest median (ties by 90th pct)."
  best, best_key = 0, None
  for i, row in enumerate(E):
    d    = {c.txt: row[c.at] for c in test.x}   # inputs only: the response stays unseen
    est  = guess(train_csv, d)[resp.txt]         # (median, 10th pct, 90th pct)
    key  = (est[0], est[2])
    if best_key is None or key > best_key: best, best_key = i, key
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
  "Ridge least squares: beta minimising |X beta - y|^2 + lam |beta|^2, intercept (first term) nearly unpenalised."
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
  for i in range(p): A[i][i] += (1e-9 if i == 0 else lam)
  return _solve(A, b)

def _features(train):
  "A feature maker from TRAIN: an intercept, standardised numeric inputs, one-hot low-cardinality symbols."
  spec = []
  for c in train.x:
    if type(c) is ezr.Sym:
      levels = sorted(c.has, key=str)
      if 0 < len(levels) <= MAX_LEVELS:
        spec += [("sym", c.at, lvl, None) for lvl in levels]
    else:
      spec.append(("num", c.at, c.mu, c.sd if c.sd > 0 else 1.0))

  def feats(row):
    out = [1.0]
    for kind, at, a, b in spec:
      v = row[at]
      if kind == "num":
        out.append(((v if _is_num(v) else a) - a) / b)
      else:
        out.append(1.0 if v == a else 0.0)
    return out
  return feats

def _ols_pick(train_csv, test, E, resp) -> int:
  "Off-the-shelf model: a linear regression of the response on the inputs; pick the customer predicted most likely to respond."
  train    = ezr.Tbl(ezr.csv(train_csv))
  tr_resp  = _response_col(train)
  known    = [r for r in train.rows if _is_num(r[tr_resp.at])]
  if not known: raise ValueError("no training row has a known response")
  feats    = _features(train)
  X        = [feats(r) for r in known]
  y        = [float(r[tr_resp.at]) for r in known]
  beta     = _ridge(X, y, RIDGE)
  best, best_p = 0, None
  for i, row in enumerate(E):
    p = sum(b * v for b, v in zip(beta, feats(row)))
    if best_p is None or p > best_p: best, best_p = i, p
  return best

def race(train_csv, test_csv) -> dict[str, float]:
  """Each method's pick on TEST_CSV, scored by wins on the response goal alone
  (100 = the best known responder, 0 = an average customer; higher is better).
  Keys: "ours" (step27 targeting), "random" (random search with the same budget),
  "highest_income" (mail the richest customer), "guesser" (step22 response guess),
  "ols" (off-the-shelf linear regression). Picks are drawn from test rows whose response is known."""
  train = ezr.Tbl(ezr.csv(train_csv))
  test  = ezr.Tbl(ezr.csv(test_csv))
  if not test.rows: raise ValueError("no test rows")
  if not test.y: raise ValueError("no goal columns (names ending in + or -)")
  resp = _response_col(test)
  E    = [r for r in test.rows if _is_num(r[resp.at])]
  if not E: raise ValueError("no test row has a known response")

  budget   = min(BUDGET, len(E))
  score    = _response_wins(E, resp)
  income   = _col_at(test, "income")
  goal_ats = {c.at for c in test.y}

  def label(i):
    "Reveal the goals of test row I, as a dict by goal name."
    return {c.txt: E[i][c.at] for c in test.y}

  runs  = {"ours": [], "random": []}
  seed0 = ezr.the.Seed
  with tempfile.TemporaryDirectory() as tmp:
    blind = os.path.join(tmp, "blind.csv")
    _write_blind(test.names, E, goal_ats, blind)
    try:
      for k in range(REPEATS):
        ezr.the.Seed = seed0 + k
        rng = random.Random(ezr.the.Seed)
        runs["ours"].append(score(E[choose(blind, label, budget, goal=resp.txt)]))
        picks = rng.sample(range(len(E)), budget)
        best  = max(picks, key=lambda i: _num_or_low(E[i][resp.at]))
        runs["random"].append(score(E[best]))
    finally:
      ezr.the.Seed = seed0

  out = {name: sum(vs) / len(vs) for name, vs in runs.items()}
  out["highest_income"] = score(E[_top_income(E, income)])
  out["guesser"]        = score(E[_guess_pick(train_csv, test, E, resp)])
  out["ols"]            = score(E[_ols_pick(train_csv, test, E, resp)])
  return out

