# step20.py
#!/usr/bin/env python3 -B
"Race our shortlist and penalty guesser against plain rivals on test rows, via ezr."
import math, os, tempfile
from collections.abc import Callable
import ezr
import step12
import step16
import step17

BUDGET = 30     # labels the shortlist may take, as in step17
LAMBDA = 1e-6   # ridge term, so the normal equations stay solvable
TOP    = 8      # most common levels one-hot encoded per symbol column

def _num(v) -> bool:
  "True for an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _raw(path: str) -> tuple[list, list]:
  "Header names and rows of PATH as dicts {name: value}; blank cells become '?'."
  src   = ezr.csv(path)
  first = next(src, None)
  if first is None: return [], []
  head  = [str(v).strip() or f"col{at}" for at, v in enumerate(first)]
  out   = []
  for r in src:
    cells = list(r) + [""] * (len(head) - len(r))
    out.append({h: ("?" if v in ("", "?") else v)
                for h, v in zip(head, cells)})
  return head, out

def _goal_score(goals: list, row: list) -> float:
  "Goal distance of ROW from the ideal, as ezr measures it; inf if a goal cell is unknown."
  if any(row[c.at] == "?" for c in goals): return math.inf
  return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in goals),
                  len(goals))

def _mean(vs: list) -> float:
  "The mean of VS, or inf if there is nothing to average."
  return sum(vs) / len(vs) if vs else math.inf

def _top_overall(full, head: list) -> int:
  "Index of the simple top overall rating: the OVR column if present, else the mean of normalised numeric inputs. Uses no labels."
  rows = full.rows
  at   = head.index("OVR") if "OVR" in head else None
  if at is not None and type(full.cols.get(at)) is ezr.Num:
    def key(i: int) -> float:
      v = rows[i][at]
      return v if _num(v) else -math.inf
  else:
    nums = [c for c in full.x if type(c) is ezr.Num]
    def key(i: int) -> float:
      vs = [ezr.norm(c, rows[i][c.at]) for c in nums if rows[i][c.at] != "?"]
      return sum(vs) / len(vs) if vs else -math.inf
  return max(range(len(rows)), key=lambda i: (key(i), -i))

def _ours(full, goals: list) -> int:
  "Index picked by step17 from a blind copy of FULL, labelling at most BUDGET rows."
  hidden = {c.at for c in goals}
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "blind.csv")
    with open(path, "w", encoding="utf-8") as f:
      f.write(",".join(str(n) for n in full.names) + "\n")
      for row in full.rows:
        f.write(",".join("?" if i in hidden else str(v)
                         for i, v in enumerate(row)) + "\n")
    def label(i: int) -> dict:
      return {c.txt: full.rows[i][c.at] for c in goals}
    return step17.choose(path, label, budget=BUDGET)

def _penalty(head: list, goals: list) -> str:
  "Name of the penalty goal: the goal whose name mentions penalties, else the first goal."
  for c in goals:
    if "penal" in head[c.at].lower(): return head[c.at]
  return head[goals[0].at]

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
        out.append(float(c.mu) if v == "?" else float(v))
    return out
  pairs = [(feats(r), float(r[target])) for r in tbl.rows if _num(r[target])]
  if not pairs: return None
  m = len(pairs[0][0])
  A = [[LAMBDA if i == j else 0.0 for j in range(m)] for i in range(m)]
  b = [0.0] * m
  for x, y in pairs:
    for i in range(m):
      xi = x[i]
      if xi == 0.0: continue
      b[i] += xi * y
      Ai = A[i]
      for j in range(m): Ai[j] += xi * x[j]
  w = _solve(A, b)
  return lambda row: sum(p * q for p, q in zip(w, feats(row)))

def race(train_csv, test_csv) -> dict[str, float]:
  """Scores on TEST_CSV; lower is better.
  Shortlist (one build picked per method, scored by its goal distance):
    'ours'         step17 labels at most BUDGET builds and picks the best.
    'top_overall'  the build with the highest OVR rating, using no labels.
  Penalty guesser (mean absolute error in training sds of the penalty goal):
    'guess_ours'   step12, the nearest-neighbour guesser.
    'guess_linear' ridge least squares on the inputs, an off-the-shelf model."""
  full, head, _ = step16._table(test_csv)
  if full is None or not full.rows: raise ValueError("the test table has no rows")
  goals = [c for c in full.y if c.txt[-1] in "+-"]
  if not goals: raise ValueError("the test table has no goal columns")

  out  = {}
  rows = full.rows
  out["ours"]         = _goal_score(goals, rows[_ours(full, goals)])
  out["top_overall"]  = _goal_score(goals, rows[_top_overall(full, head)])

  ttbl, thead, tkinds = step16._table(train_csv)
  if ttbl is None: raise ValueError("the training table has no rows")
  name = _penalty(head, goals)
  if name not in thead:
    raise ValueError(f"the training table has no {name!r} column")
  tat  = thead.index(name)
  tcol = ttbl.cols.get(tat)
  if type(tcol) is not ezr.Num:
    raise ValueError(f"the training table has no usable {name!r} goal")
  scale  = tcol.sd or 1.0
  linear = _fit_linear(ttbl, tat)

  raw_head, raw = _raw(test_csv)
  goal_names    = {h for h in raw_head if h[-1:] in ("+", "-")}
  guess_err, lin_err = [], []
  for row in raw:
    truth = row.get(name, "?")
    if not _num(truth) or not math.isfinite(truth): continue
    inputs = {k: v for k, v in row.items() if k not in goal_names}
    est    = step12.guess(train_csv, inputs).get(name)
    if est is not None:
      guess_err.append(abs(est[0] - truth) / scale)
    if linear is not None:
      pred = linear(step16._align(thead, tkinds, row))
      lin_err.append(abs(pred - truth) / scale)

  out["guess_ours"]   = _mean(guess_err)
  out["guess_linear"] = _mean(lin_err)
  return out

