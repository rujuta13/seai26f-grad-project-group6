# step9.py: spot. Flag benchmark rows that look like glitches or are unlike anything seen.
"step9.py: spot. Flag benchmark rows that look like glitches or are unlike anything seen."
import functools
import math
import ezr

SAMPLE = 100   # training rows used to measure the normal nearest-neighbour gap
Q      = 0.95  # the normal gap is this quantile of leave-one-out gaps
SIGMA  = 3     # a number this many sds from train's mean is extreme
NEW    = 1.0   # an unseen symbol, or a non-number in a number column, counts this much

def _val(v):
  "Cells from dicts may still be strings: coerce them the way csv does."
  return ezr.atom(v) if isinstance(v, str) else v

def _align(tbl, row):
  "ROW in TBL's column order, from a dict by column name or from a list."
  if isinstance(row, dict):
    return [_val(row.get(n, "?")) for n in tbl.names]
  return [_val(v) for v in row]

def _dist(cs, a, b):
  "Root mean square of the column gaps between full rows A and B."
  return ezr.dist((ezr.gap(c, a[c.at], b[c.at]) for c in cs), len(cs))

def _extreme(c, v):
  "How far one cell V is from what its column shows in train; 0 if unknown or typical."
  if v == "?": return 0.0
  if type(c) is ezr.Sym:
    return 0.0 if v in c.has else NEW        # a symbol never seen in train
  if not isinstance(v, (int, float)):
    return NEW                               # text where a number belongs
  if c.sd == 0:
    return 0.0 if v == c.mu else NEW         # train never varies here
  z = abs(v - c.mu) / c.sd
  return max(0.0, z - SIGMA)                 # only the far tail counts

@functools.lru_cache(maxsize=8)
def _model(train_csv):
  "The table, its columns, its full rows, and the normal nearest-neighbour gap."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  cs  = [tbl.cols[at] for at in sorted(tbl.cols)]
  vs  = tbl.rows
  n   = len(vs)
  if n < 2: return tbl, cs, vs, 1.0
  step = max(1, n // SAMPLE)
  base = sorted(min((_dist(cs, vs[i], vs[j]) for j in range(n) if j != i),
                    default=math.inf)
                for i in range(0, n, step))
  scale = base[min(len(base) - 1, int(Q * len(base)))]
  return tbl, cs, vs, (scale or 1.0)       # zero gap in train: use 1 as the unit

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row of ROWS (goals included): higher is more glitch-like or unlike train."
  tbl, cs, vs, scale = _model(train_csv)
  out = []
  for row in rows:
    r    = _align(tbl, row)
    near = min((_dist(cs, r, t) for t in vs), default=math.inf)
    ext  = sum(_extreme(c, r[c.at]) for c in cs)
    out.append(float(near / scale + ext))
  return out

