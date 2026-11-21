# step29.py: spot. Flag customer records that look wrong, or are unlike anyone else.
"step29.py: spot. Flag customer records that look wrong, or are unlike anyone else."
import functools
import math
import ezr

SAMPLE = 100   # training rows used to measure the normal nearest-neighbour gap
Q      = 0.95  # the normal gap is this quantile of leave-one-out gaps
SIGMA  = 3     # a number this many sds from train's mean is extreme
NEW    = 1.0   # an unseen symbol, or a non-number where a number belongs, counts this much

def _is_num(v) -> bool:
  "True if V is a usable finite number (bools and infinities do not count)."
  return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _val(v):
  "One raw cell as an atom: blanks, None and NaN become '?'; strings are coerced like csv does."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return "?" if s == "" else ezr.atom(s)
  if isinstance(v, float) and math.isnan(v): return "?"
  return v

def _clean(tbl, row):
  """ROW in TBL's column order, goals included, each cell coerced. A dict may omit a
  column, or key a goal without its suffix; missing cells are '?'."""
  n = len(tbl.names)
  if isinstance(row, dict):
    raw = []
    for name in tbl.names:
      key = str(name)
      if key in row:                                     raw.append(row[key])
      elif key and key[-1] in "+-!" and key[:-1] in row: raw.append(row[key[:-1]])
      else:                                              raw.append("?")
  else:
    raw  = list(row)[:n]
    raw += ["?"] * (n - len(raw))
  return [_val(v) for v in raw]

def _vec(cs, row):
  "ROW as one entry per column: 0..1 for numbers, the symbol for symbols, None if unknown or not a number."
  out = []
  for c in cs:
    v = row[c.at]
    if v == "?":
      out.append(None)
    elif type(c) is ezr.Sym:
      out.append(v)
    elif not _is_num(v):
      out.append(None)
    else:
      out.append(ezr.norm(c, v))
  return out

def _dist(cs, a, b):
  "Root mean square of the column gaps between two _vecs; unknown is far, symbols match or not."
  if not cs: return 0.0
  total = 0.0
  for c, x, y in zip(cs, a, b):
    if x is None or y is None: g = 1.0
    elif type(c) is ezr.Sym:   g = 0.0 if x == y else 1.0
    else:                      g = x - y
    total += g * g
  return math.sqrt(total / len(cs))

def _extreme(c, v):
  "How far one cleaned cell V is from what its column shows in train; 0 if unknown or typical."
  if v == "?": return 0.0
  if type(c) is ezr.Sym:
    return 0.0 if v in c.has else NEW        # a value never seen in train
  if not _is_num(v): return NEW              # text or junk where a number belongs
  if c.n == 0: return NEW                    # train has no numbers in this column
  if c.sd == 0:
    return 0.0 if v == c.mu else NEW         # train never varies here
  z = abs(v - c.mu) / c.sd
  return max(0.0, z - SIGMA)                 # only the far tail counts

@functools.lru_cache(maxsize=8)
def _model(train_csv):
  "The table, its columns, its rows as _vecs, and the normal nearest-neighbour gap."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  if not tbl.rows: raise ValueError("train has no rows to compare against")
  cs  = [tbl.cols[at] for at in sorted(tbl.cols)]
  vs  = [_vec(cs, r) for r in tbl.rows]
  n   = len(vs)
  if n < 2: return tbl, cs, vs, 1.0
  step = max(1, n // SAMPLE)
  base = sorted(min(_dist(cs, vs[i], vs[j]) for j in range(n) if j != i)
                for i in range(0, n, step))
  scale = base[min(len(base) - 1, int(Q * len(base)))]
  if scale <= 0: scale = 1.0                 # zero gap in train: use 1 as the unit
  return tbl, cs, vs, scale

def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row of ROWS (goals included): higher is more glitch-like or unlike train."
  tbl, cs, vs, scale = _model(train_csv)
  out = []
  for row in rows:
    r    = _clean(tbl, row)
    v    = _vec(cs, r)
    near = min((_dist(cs, v, u) for u in vs), default=math.inf)
    ext  = sum(_extreme(c, r[c.at]) for c in cs)
    out.append(float(near / scale + ext))
  return out

