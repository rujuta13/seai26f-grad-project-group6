# step11.py
import ezr

K   = 5      # how many families of player: a handful
TOP = 0.05   # best: at most this share of rows


def _is_num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _goals(tbl: ezr.TBL) -> list:
  "The numeric goal columns of TBL: names ending in + or -."
  return [c for c in tbl.cols.values()
          if type(c) is ezr.Num and tbl.names[c.at][-1] in "+-"]

def _ydist(goals: list, r: ezr.ROW) -> float:
  "How far R's goals are from the best they could be (0 is best)."
  return ezr.dist((abs(ezr.norm(c, r[c.at]) - c.goal) for c in goals),
                  len(goals))

def _usable(tbl: ezr.TBL, cols: list) -> list:
  "Input columns worth clustering on. Ids and names are nearly unique symbols, so they add only noise."
  n = len(tbl.rows)
  return [c for c in cols
          if type(c) is ezr.Num or len(c.has) <= max(1, n // 2)]

def _encode(cols: list, row: ezr.ROW) -> list:
  "ROW over COLS: a normalised number for a Num, the symbol for a Sym, None if unknown."
  out = []
  for c in cols:
    v = row[c.at]
    if v is None or (isinstance(v, str) and v.strip() in ("", "?")):
      out.append(None)
    elif type(c) is ezr.Num:
      out.append(ezr.norm(c, v) if _is_num(v) else None)
    else:
      out.append(v)
  return out

def _far(r: list, m: list, num: list) -> float:
  "Squared distance between encoded row R and encoded centroid M (same ranking as the root mean square)."
  total = 0.0
  for a, b, n in zip(r, m, num):
    if a is None or b is None: g = 1.0          # unknown = far
    elif n:                    g = abs(a - b)
    else:                      g = 0.0 if a == b else 1.0
    total += g * g
  return total

def _centre(rows: list, num: list) -> list:
  "Middle of encoded ROWS: the mean for numbers, the most common symbol otherwise; None if none known."
  out = []
  for j, n in enumerate(num):
    vals = [r[j] for r in rows if r[j] is not None]
    if not vals:
      out.append(None)
    elif n:
      out.append(sum(vals) / len(vals))
    else:
      counts: dict = {}
      for v in vals: counts[v] = counts.get(v, 0) + 1
      out.append(max(counts, key=counts.get))
  return out


def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  if not rows: return []
  cols = _usable(tbl, tbl.x)
  if not cols: return [0] * len(rows)
  num  = [type(c) is ezr.Num for c in cols]
  enc  = [_encode(cols, r) for r in rows]
  k    = min(K, len(rows))
  # farthest-point seeding: deterministic, and spreads the seeds out
  seeds = [0]
  dmin  = [_far(e, enc[0], num) for e in enc]
  while len(seeds) < k:
    far   = max(range(len(enc)), key=lambda i: dmin[i])
    seeds.append(far)
    dmin  = [min(d, _far(e, enc[far], num)) for d, e in zip(dmin, enc)]
  cents = [enc[i] for i in seeds]
  # k-modes style refinement: assign to nearest, then recentre on middles
  labels: list[int] = []
  for _ in range(30):
    new = [min(range(k), key=lambda j: _far(e, cents[j], num)) for e in enc]
    if new == labels: break
    labels = new
    for j in range(k):
      members = [e for e, l in zip(enc, labels) if l == j]
      if members: cents[j] = _centre(members, num)
  return labels

def best(train_csv) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  rows  = tbl.rows
  goals = _goals(tbl)
  cap   = int(TOP * len(rows))
  if not goals or cap < 1: return []
  scored = []
  for i, r in enumerate(rows):
    if all(_is_num(r[c.at]) for c in goals):   # incomplete goals cannot be ranked
      scored.append((_ydist(goals, r), i))
  scored.sort()
  return [i for _, i in scored[:cap]]

