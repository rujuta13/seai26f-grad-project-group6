# step11.py
"Step 11: remember. Sort players into families, and find the unusually good ones (strong and good from the spot = best on all goals). Results are remembered per file."
import math
import os
import random
import ezr

K     = 5             # number of families to find
ITERS = 20            # cap on k-modes iterations
SEED  = 1234567891    # fixed seed, so results repeat
FRAC  = 0.05          # "best" is at most this fraction of rows

_CACHE: dict = {}     # (path, mtime) -> (families, best), remembered per file

def _key(train_csv: str) -> tuple:
  "Identity of TRAIN_CSV: its absolute path and when it last changed."
  path = os.path.abspath(os.path.expanduser(train_csv))
  return (path, os.path.getmtime(path))

def _load(train_csv: str):
  "Table of TRAIN_CSV, with columns and goals from its header."
  return ezr.Tbl(ezr.csv(train_csv))

def _inputs(tbl) -> list:
  """Input columns that carry family information: all numbers, and symbols
  that repeat. A symbol where every value is distinct is an identifier
  (like a row number or url), so it says nothing about families."""
  return [c for c in tbl.x
          if type(c) is ezr.Num or len(c.has) < c.n]

def _x(c, v):
  "One input cell as a feature: numbers become 0..1 (None if unknown); symbols stay as they are."
  if type(c) is ezr.Num:
    if isinstance(v, (int, float)) and math.isfinite(v):
      return ezr.norm(c, v)
    return None
  return None if v == "?" else v

def _gap2(cols: list, f: list, g: list) -> float:
  "Squared distance between feature vectors F and G, one gap per input column. Unknown = far."
  total = 0.0
  for c, a, b in zip(cols, f, g):
    if a is None or b is None: d = 1.0
    elif type(c) is ezr.Num:   d = abs(a - b)
    else:                      d = 0.0 if a == b else 1.0
    total += d * d
  return total

def _centre(cols: list, members: list) -> list:
  "Middle of MEMBERS: the mean for numbers, the most common value for symbols; None if unknown."
  out = []
  for i, c in enumerate(cols):
    vals = [f[i] for f in members if f[i] is not None]
    if not vals:
      out.append(None)
    elif type(c) is ezr.Num:
      out.append(sum(vals) / len(vals))
    else:
      count: dict = {}
      for v in vals: count[v] = count.get(v, 0) + 1
      out.append(max(sorted(count, key=str), key=lambda v: count[v]))
  return out

def _families(tbl) -> list[int]:
  "One family label per row of TBL, in row order: k-modes over the input columns."
  rows = tbl.rows
  n    = len(rows)
  if not n: return []
  cols  = _inputs(tbl)
  feats = [[_x(c, r[c.at]) for c in cols] for r in rows]
  k     = min(K, n)
  rng   = random.Random(SEED)
  cents = [feats[i] for i in rng.sample(range(n), k)]  # seed with rows
  labels = [-1] * n
  for _ in range(ITERS):
    # assign each row to its nearest centroid, over the input columns
    new = [min(range(k), key=lambda j, f=f: _gap2(cols, f, cents[j]))
           for f in feats]
    if new == labels: break
    labels = new
    # move each centroid to the middle of its members (empty ones stay put)
    for j in range(k):
      members = [f for f, lab in zip(feats, labels) if lab == j]
      if members: cents[j] = _centre(cols, members)
  return labels

def _best(tbl) -> list[int]:
  """Row indices of TBL's unusually good rows: closest to the ideal on all
  numeric goals together, at most FRAC of rows, best first. Rows with no
  known goal value go last."""
  rows  = tbl.rows
  goals = [c for c in tbl.y if type(c) is ezr.Num]
  if not goals: return []
  n = int(FRAC * len(rows))
  def score(r) -> float:
    sq, k = 0.0, 0
    for c in goals:
      v = r[c.at]
      if isinstance(v, (int, float)) and math.isfinite(v):
        sq += (ezr.norm(c, v) - c.goal) ** 2
        k  += 1
    return math.sqrt(sq / k) if k else math.inf
  ds    = [score(r) for r in rows]
  order = sorted(range(len(rows)), key=lambda i: (ds[i], i))
  return order[:n]

def _remember(train_csv: str) -> tuple:
  "Families and best for TRAIN_CSV, computed once per file version, then reused."
  key = _key(train_csv)
  if key not in _CACHE:
    tbl = _load(train_csv)
    _CACHE[key] = (_families(tbl), _best(tbl))
  fam, bst = _CACHE[key]
  return list(fam), list(bst)

def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  return _remember(train_csv)[0]

def best(train_csv) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows, best first."
  return _remember(train_csv)[1]

