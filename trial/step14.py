# step14.py: flow. Keep the good rows as they stream in; warn when they drift.
"step14.py: flow. Keep the good rows as they stream in; warn when they drift."
import collections
import functools
import math
import ezr

GOOD_PCT = 25     # a row is worth keeping if its goal distance is in the best 25% of training
Q        = 0.95   # the normal gap is this quantile of leave-one-out nearest-neighbour gaps
WIN      = 40     # how many recent rows drift looks at
NOVEL    = 10     # warn when this many of the last WIN rows are novel
SAMPLE   = 100    # training rows used to measure the normal gap

def _val(v):
  "Cells from dicts may still be strings: coerce them the way csv does."
  return ezr.atom(v) if isinstance(v, str) else v

def _align(tbl, row):
  "ROW in TBL's column order, from a dict by column name or from a list; missing cells are '?'."
  if isinstance(row, dict):
    return [_val(row.get(n, "?")) for n in tbl.names]
  return [_val(v) for v in row]

def _ydist(tbl, row):
  "How far ROW's goals are from ideal; infinite if there are no goals or any goal is unknown."
  if not tbl.y: return math.inf
  if any(row[c.at] == "?" for c in tbl.y): return math.inf
  return ezr.ydist(tbl, row)

@functools.lru_cache(maxsize=8)
def _keep_model(train_csv):
  "The table, and the ydist at or below which a row is worth keeping (best GOOD_PCT of train)."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  known = sorted(y for y in (_ydist(tbl, r) for r in tbl.rows) if y < math.inf)
  cap   = max(1, len(known) * GOOD_PCT // 100)
  cut   = known[min(cap, len(known)) - 1] if known else -math.inf
  return tbl, cut

def _vec(cs, row):
  "ROW as one entry per column: 0..1 for nums, the symbol for syms; None if unknown or not a number."
  out = []
  for c in cs:
    v = row[c.at]
    if v == "?":
      out.append(None)
    elif type(c) is ezr.Sym:
      out.append(v)
    elif not isinstance(v, (int, float)):
      out.append(None)
    else:
      out.append(ezr.norm(c, v))
  return out

def _gap2(cs, a, b):
  "Squared distance between two _vecs, column by column; unknown is far."
  total = 0
  for c, x, y in zip(cs, a, b):
    if x is None or y is None: g = 1
    elif type(c) is ezr.Sym:   g = 0 if x == y else 1
    else:                      g = abs(x - y)
    total += g * g
  return total

@functools.lru_cache(maxsize=8)
def _drift_model(train_csv):
  """The table, its columns, its rows as _vecs, and the normal gap:
  a high quantile of leave-one-out nearest-neighbour gaps within train."""
  tbl = ezr.Tbl(ezr.csv(train_csv))
  cs  = [tbl.cols[at] for at in sorted(tbl.cols)]
  vs  = [_vec(cs, r) for r in tbl.rows]
  if len(vs) < 2: return tbl, cs, vs, math.inf
  step = max(1, len(vs) // SAMPLE)
  base = sorted(min(_gap2(cs, vs[i], vs[j]) for j in range(len(vs)) if j != i)
                for i in range(0, len(vs), step))
  return tbl, cs, vs, base[min(len(base) - 1, int(Q * len(base)))]

def stream(train_csv, rows) -> list[bool]:
  """For each of ROWS (arriving one at a time), True if it is worth keeping:
  its goals are in the best GOOD_PCT of train_csv's rows. Each decision uses only
  train_csv and that row's own cells, never any later row."""
  tbl, cut = _keep_model(train_csv)
  return [_ydist(tbl, _align(tbl, row)) <= cut for row in rows]

def drift(train_csv, rows) -> int | None:
  """Index into ROWS of the row at which the recent rows first stop looking
  like train_csv (too many nearest-neighbour gaps beyond normal), or None.
  Uses only train_csv and the rows up to and including the one being judged."""
  tbl, cs, vs, cut = _drift_model(train_csv)
  recent = collections.deque(maxlen=WIN)
  for i, row in enumerate(rows):
    v    = _vec(cs, _align(tbl, row))
    near = min((_gap2(cs, v, u) for u in vs), default=math.inf)
    recent.append(near > cut)
    if sum(recent) >= NOVEL: return i
  return None

