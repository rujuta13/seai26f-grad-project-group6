# step30.py
#!/usr/bin/env python3 -B
"Race our targeting and response guesser against plain rivals on test rows, via ezr."
import math, os, random, tempfile
from collections.abc import Callable
import ezr
import step16
import step22
import step27

BUDGET = 50            # labels each targeting method may take: a test mailing of 50
LAMBDA = 1e-6          # ridge term, so the normal equations stay solvable
TOP    = 8             # most common levels one-hot encoded per symbol column
TARGET = "RESPONSE+"   # the goal the response guesser is raced on

def _num(v) -> bool:
  "True for an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _mean(vs: list) -> float:
  "The mean of VS, or inf if there is nothing to average."
  return sum(vs) / len(vs) if vs else math.inf

def _blind(full, goals, path: str) -> None:
  "Write FULL to PATH with every goal cell '?', so a method must label to see them."
  hidden = {c.at for c in goals}
  with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(str(n) for n in full.names) + "\n")
    for row in full.rows:
      f.write(",".join("?" if i in hidden else str(v)
                       for i, v in enumerate(row)) + "\n")

def _yardstick(full, goals, i: int) -> float:
  "Goal distance of row I from the ideal, as the evaluation yardstick; inf if a goal is unknown."
  row = full.rows[i]
  if any(not _num(row[c.at]) for c in goals): return math.inf
  return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in goals),
                  len(goals))

def _best_known(goals, known: dict) -> int:
  "Index in KNOWN (index -> {col at: goal cell}) nearest the ideal, judged by its labels alone."
  cols = {c.at: ezr.Num(c.txt, c.at) for c in goals}
  for cells in known.values():
    for c in goals:
      if _num(cells[c.at]): ezr.add(cols[c.at], cells[c.at])
  def d(i: int) -> float:
    cells = known[i]
    if any(not _num(cells[c.at]) for c in goals): return math.inf
    return ezr.dist((abs(ezr.norm(cols[c.at], cells[c.at]) - cols[c.at].goal)
                     for c in goals), len(goals))
  return min(known, key=lambda i: (d(i), i))

def _income_at(names) -> int:
  "Position of the INCOME column in NAMES."
  for at, n in enumerate(names):
    if str(n).strip().upper() == "INCOME": return at
  raise ValueError("the test table has no INCOME column")

def _top_income(full) -> int:
  "Index of the highest-income row; uses no labels. Ties keep file order."
  at   = _income_at(full.names)
  rows = full.rows
  def key(i: int) -> float:
    v = rows[i][at]
    return v if _num(v) and math.isfinite(v) else -math.inf
  return max(range(len(rows)), key=lambda i: (key(i), -i))

def _solve(A: list, b: list) -> list[float]:
  "Solve A w = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [row[:] + [b[i]] for i, row in enumerate(A)]
  for k in range(n):
    p = max(range(k, n), key=lambda r: abs(M[r][k]))
    M[k], M[p] = M[p], M[k]
    pv = M[k][k]
    if abs(pv) < 1e-12: continue
    M[k] = [v / pv for v in M[k]]
    for r in range(n):
      if r != k and M[r][k] != 0:
        f = M[r][k]
        M[r] = [a - f * v for a, v in zip(M[r], M[k])]
  return [M[i][n] for i in range(n)]

def _fit_linear(tbl, target: int) -> Callable[[list], float] | None:
  "Ridge least squares of column TARGET on the inputs of TBL: symbols one-hot (most common levels), numbers as is."
  xs     = list(tbl.x)
  levels = {c.at: sorted(c.has, key=lambda v: (-c.has[v], str(v)))[:TOP]
            for c in xs if type(c) is ezr.Sym}
  def feats(row: list) -> list[float]:
    out = [1.0]                                    # intercept
    for c in xs:
      v = row[c.at]
      if type(c) is ezr.Sym:
        out += [1.0 if v == lv else 0.0 for lv in levels[c.at]]
      else:
        out.append(float(v) if _num(v) else float(c.mu))
    return out
  pairs = [(feats(r), float(r[target])) for r in tbl.rows if _num(r[target])]
  if not pairs: return None
  m = len(pairs[0][0])
  A = [[LAMBDA if i == j else 0.0 for j in range(m)] for i in range(m)]
  b = [0.0] * m
  for x, y in pairs:
    for i in range(m):
      b[i] += x[i] * y
      Ai = A[i]
      for j in range(m): Ai[j] += x[i] * x[j]
  w = _solve(A, b)
  return lambda row: sum(p * q for p, q in zip(w, feats(row)))

def _race_targeting(full, goals, rng: random.Random) -> dict[str, float]:
  "Goal distance of the customer each targeting method picks, over every goal."
  n      = len(full.rows)
  budget = min(BUDGET, n)
  def label(i: int) -> dict:
    return {c.txt: full.rows[i][c.at] for c in goals}
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "blind.csv")
    _blind(full, goals, path)
    ours = step27.choose(path, label, budget=budget)
  random_known = {i: {c.at: full.rows[i][c.at] for c in goals}
                  for i in rng.sample(range(n), budget)}
  picks = {
    "ours":          ours,
    "top_income":    _top_income(full),
    "random_search": _best_known(goals, random_known),
  }
  return {k: _yardstick(full, goals, i) for k, i in picks.items()}

def _race_guessing(train_csv, full, goals) -> dict[str, float]:
  "Mean scaled absolute error of each response guesser on TARGET, over test rows."
  names = [str(n) for n in full.names]
  if TARGET not in names:
    raise ValueError(f"the test table has no {TARGET!r} column")
  at_t = names.index(TARGET)
  ttbl, thead, tkinds = step16._table(train_csv)
  if ttbl is None: raise ValueError("the training table has no rows")
  if TARGET not in thead:
    raise ValueError(f"the training table has no {TARGET!r} column")
  tat  = thead.index(TARGET)
  tcol = ttbl.cols.get(tat)
  if type(tcol) is not ezr.Num:
    raise ValueError(f"the training table has no usable {TARGET!r} goal")
  scale  = tcol.sd or 1.0
  linear = _fit_linear(ttbl, tat)
  goal_names = {c.txt for c in goals}
  errs = {"guess_ours": [], "guess_linear": []}
  for row in full.rows:
    truth = row[at_t]
    if not _num(truth) or not math.isfinite(truth): continue
    raw = {name: v for name, v in zip(names, row) if name not in goal_names}
    est = step22.guess(train_csv, raw).get(TARGET)
    if est is not None:
      errs["guess_ours"].append(abs(est[0] - truth) / scale)
    if linear is not None:
      pred = linear(step16._align(thead, tkinds, raw))
      errs["guess_linear"].append(abs(pred - truth) / scale)
  return {k: _mean(v) for k, v in errs.items()}

def race(train_csv, test_csv) -> dict[str, float]:
  """Scores on TEST_CSV; lower is better.
  Targeting (goal distance of the customer picked, over every goal):
    'ours'          step27, labelling at most BUDGET customers and picking the best.
    'top_income'    the customer with the highest INCOME, using no labels.
    'random_search' BUDGET customers labelled at random, the best one kept.
  Response guesser (mean absolute error in training sds of RESPONSE+):
    'guess_ours'    step22, the nearest-neighbour guesser.
    'guess_linear'  ridge least squares on the inputs, an off-the-shelf model."""
  full  = ezr.Tbl(ezr.csv(test_csv))
  if not full.rows: raise ValueError("the test table has no rows")
  goals = [c for c in full.y if c.txt[-1] in "+-"]
  if not goals: raise ValueError("the test table has no goal columns")
  rng = random.Random(ezr.the.Seed)
  out = _race_targeting(full, goals, rng)
  out.update(_race_guessing(train_csv, full, goals))
  return out

