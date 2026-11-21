# step20.py
"Step 20: race. Does our shortlist beat a simple top-overall-rating list, and does our penalty guesser beat an off-the-shelf model? All scores are lower-is-better."
import math
import random
import ezr
import step12
import step17

BUDGET = 30           # labels each shortlist method may spend
PEN    = "Penalties"  # the goal the penalty guesser predicts
RIDGE  = 1.0          # penalty on the linear model's weights
MINLEV = 3            # a symbol level must occur this often to get its own feature
SEED   = 1234567891   # fixed seed, so results repeat

def _load(csv_path: str):
  "Table of CSV_PATH, with columns and goals from its header."
  return ezr.Tbl(ezr.csv(csv_path))

def _name(s: str) -> str:
  "A column name without its +, - or ! suffix."
  return s[:-1] if s[-1:] in ("+", "-", "!") else s

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _sd(c) -> float:
  "A column's standard deviation, or 1.0 if it has none."
  return c.sd if c.sd > 0 else 1.0

def _mean(xs: list) -> float:
  "Mean of XS, or infinity if XS is empty."
  return sum(xs) / len(xs) if xs else math.inf

def _dist(gtr, vals: dict) -> float:
  "Distance of goal values VALS (goal name -> value) from the ideal, scaled by train goals GTR."
  sq, n = 0.0, 0
  for c in gtr:
    v = vals.get(c.txt, "?")
    if not _isnum(v): continue
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

def _shortlist(test_csv: str, gtr) -> dict[str, float]:
  "Score our shortlist (step17.choose), the top-OVR list, and random picks, all with the same label budget."
  tbl  = _load(test_csv)
  rows = tbl.rows
  n    = len(rows)
  if n == 0: raise ValueError("race: no builds in test_csv")
  budget = min(BUDGET, n)

  def label(i: int) -> dict:
    "The true goal values of row I: the oracle, billed once per call."
    return {c.txt: rows[i][c.at] for c in tbl.y}

  # ours: our guided choice
  ours_i = step17.choose(test_csv, label, budget=budget)
  ours   = _dist(gtr, label(ours_i))

  # top overall rating: label the BUDGET highest-OVR rows, keep the best of them
  ovr = [c for c in tbl.x if _name(c.txt) == "OVR"]
  if not ovr: raise ValueError("race: no OVR column to rank by")
  at = ovr[0].at

  def top(i: int) -> float:
    v = rows[i][at]
    return v if _isnum(v) else -math.inf

  order       = sorted(range(n), key=lambda i: (-top(i), i))
  top_overall = min(_dist(gtr, label(i)) for i in order[:budget])

  # random: BUDGET random rows, keep the best of them
  picks  = random.Random(SEED).sample(range(n), budget)
  random_best = min(_dist(gtr, label(i)) for i in picks)

  return {"ours": ours, "top_overall": top_overall, "random": random_best}

def _penalty(train_csv: str, test_csv: str) -> dict[str, float]:
  "Score our penalty guesser (step12.guess), a ridge regression, and the train mean, on the same test builds."
  ttr = _load(train_csv)
  tst = _load(test_csv)

  tgt = [c for c in ttr.y if type(c) is ezr.Num and _name(c.txt) == PEN]
  if not tgt: raise ValueError(f"race: no numeric {PEN} goal in train")
  pc  = tgt[0]
  tp  = [c for c in tst.y if c.txt == pc.txt]
  if not tp: raise ValueError(f"race: no {pc.txt} goal in test")
  pat = tp[0].at
  sd  = _sd(pc)

  truths = [(row, row[pat]) for row in tst.rows if _isnum(row[pat])]
  if not truths: raise ValueError("race: no test build has a penalty value")

  # ours: step12's neighbour median, from input columns only
  errs = []
  for row, truth in truths:
    inputs = {c.txt: row[c.at] for c in tst.x}
    est    = step12.guess(train_csv, inputs)
    if pc.txt not in est: continue
    e = est[pc.txt][0]
    if math.isnan(e): continue
    errs.append(abs(e - truth) / sd)
  ours = _mean(errs)

  # rival: the train mean, predicted for every build
  mu       = pc.mu
  mean_err = _mean([abs(mu - t) / sd for _, t in truths])

  # rival: ridge regression on standardised numbers and frequent symbol levels
  fit = [r for r in ttr.rows if _isnum(r[pc.at])]
  if not fit: raise ValueError("race: no train build has a penalty value")
  feat_cols = [c for c in ttr.x
               if type(c) is ezr.Num or len(c.has) < c.n]
  levels = {c.at: sorted((v for v, k in c.has.items() if k >= MINLEV),
                         key=str)
            for c in feat_cols if type(c) is ezr.Sym}

  def feats(row: list) -> list:
    "Full-width ROW into a feature vector, with a leading intercept."
    f = [1.0]
    for c in feat_cols:
      v = row[c.at]
      if type(c) is ezr.Num:
        f.append((v - c.mu) / _sd(c) if _isnum(v) else 0.0)
      else:
        f.extend(1.0 if v == s else 0.0 for s in levels[c.at])
    return f

  X = [feats(r) for r in fit]
  Y = [(r[pc.at] - mu) / sd for r in fit]
  p = len(X[0])
  A = [[0.0] * p for _ in range(p)]
  b = [0.0] * p
  for x, y in zip(X, Y):
    for i, xi in enumerate(x):
      if xi == 0.0: continue
      row_a = A[i]
      for j, xj in enumerate(x):
        row_a[j] += xi * xj
      b[i] += xi * y
  for i in range(p): A[i][i] += RIDGE
  w = [r[0] for r in _solve(A, [[bi] for bi in b])]

  errs = []
  for row, truth in truths:
    f    = feats(row)
    pred = sum(fi * wi for fi, wi in zip(f, w))
    errs.append(abs(pred - (truth - mu) / sd))
  linear = _mean(errs)

  return {"penalty_ours": ours, "penalty_linear": linear,
          "penalty_mean": mean_err}

def race(train_csv, test_csv) -> dict[str, float]:
  """Race our shortlist (step17) against top-OVR and random picks, and our
  penalty guesser (step12) against ridge regression and the train mean, on
  test_csv. Returns {method: score}, including the key "ours" (our shortlist).
  Shortlist scores are the goal distance of the best build found (0 is ideal).
  Penalty scores are the mean penalty error in train standard deviations.
  All scores are lower-is-better."""
  gtr = [c for c in _load(train_csv).y if type(c) is ezr.Num]
  if not gtr: raise ValueError("race: no numeric goal to race on")
  out = _shortlist(test_csv, gtr)
  out.update(_penalty(train_csv, test_csv))
  return {k: float(v) for k, v in out.items()}

