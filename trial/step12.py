# step12.py
import math
import ezr

LO, HI = 0.10, 0.90   # the band we are fairly sure of: the middle 80% of neighbours
MIN_K  = 5            # fewest neighbours that count as evidence

_cache: dict[str, ezr.TBL] = {}   # load each training file once

def _load(train_csv: str) -> ezr.TBL:
  "Load TRAIN_CSV, and remember it for later calls."
  if train_csv not in _cache:
    _cache[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _cache[train_csv]

def _is_num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _cell(v):
  "A raw cell as an ezr atom; None, blank and '?' become '?'."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return "?" if s in ("", "?") else ezr.atom(s)
  return v

def _align(tbl: ezr.TBL, row) -> ezr.ROW:
  """Align ROW to TBL's columns. ROW is a dict by column name (with or
  without its + or - sign), or a list over every column or over the
  non-goal columns. Missing is '?'."""
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name, row.get(name.rstrip("+-"))))
             for at, name in enumerate(tbl.names)]
  elif len(row) == len(tbl.names):
    pairs = list(enumerate(row))
  else:
    keep  = [at for at, n in enumerate(tbl.names) if n[-1] not in "+-"]
    pairs = list(zip(keep, row))
  for at, v in pairs: q[at] = _cell(v)
  return q

def _quantile(xs: list[float], p: float) -> float:
  "The value at fraction P of the sorted, non-empty list XS, interpolated between ranks."
  pos = p * (len(xs) - 1)
  lo  = math.floor(pos)
  hi  = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

def _values(rows: ezr.ROWS, at: int) -> list[float]:
  "The sorted numeric values in column AT of ROWS."
  return sorted(float(r[at]) for r in rows if _is_num(r[at]))

def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """Per goal: (estimate, low, high) for ROW, whose goal columns are absent.
  The estimate is the median goal value of the training rows nearest to ROW
  over the input columns. Low..high is the middle 80% of those neighbours'
  values: the range we are fairly sure the true value falls in. A wide band
  means low confidence; a narrow one means the neighbours agree."""
  tbl   = _load(train_csv)
  names = tbl.names
  goals = [c for c in tbl.y if type(c) is ezr.Num]
  if not goals: return {}
  q = _align(tbl, row)
  for c in goals: q[c.at] = "?"        # goals are never inputs
  if not tbl.rows:
    return {names[c.at]: (0.0, 0.0, 0.0) for c in goals}
  if any(q[c.at] != "?" for c in tbl.x):
    pool = sorted(tbl.rows, key=lambda r: ezr.xdist(tbl, q, r))
  else:                                # no input known: every row is a neighbour
    pool = list(tbl.rows)
  k    = min(len(pool), max(MIN_K, int(math.sqrt(len(tbl.rows)))))
  near = pool[:k]
  out  = {}
  for c in goals:
    vals = _values(near, c.at)
    if len(vals) < MIN_K:              # too few neighbours know this goal
      vals = _values(tbl.rows, c.at)
    if not vals:                       # nothing known anywhere for this goal
      out[names[c.at]] = (0.0, 0.0, 0.0)
      continue
    est = _quantile(vals, 0.5)
    lo  = min(_quantile(vals, LO), est)
    hi  = max(_quantile(vals, HI), est)
    out[names[c.at]] = (est, lo, hi)
  return out

